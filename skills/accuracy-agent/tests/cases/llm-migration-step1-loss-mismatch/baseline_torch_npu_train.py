import argparse
import json
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401

from shared_case_assets import (
    JsonlLogger,
    attention_scale,
    build_causal_mask,
    build_fixed_batch,
    build_shared_weights,
    enable_alignment_determinism,
    get_case_config,
    set_global_seed,
    softmax_entropy,
    summarize_named_array,
    summarize_arrays,
)


def resolve_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    raise RuntimeError(
        "This baseline script requires a PyTorch torch_npu runtime with an "
        "available NPU device. CPU fallback is intentionally disabled."
    )


def to_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


class TinyCausalLM(nn.Module):
    def __init__(self, cfg, weights, compute_dtype, alignment_mode=False):
        super().__init__()
        h = cfg["hidden_size"]
        f = cfg["ffn_size"]
        v = cfg["vocab_size"]
        s = cfg["seq_len"]
        self.num_heads = cfg["num_heads"]
        self.head_dim = h // self.num_heads
        self.scale = attention_scale(cfg)
        self.compute_dtype = compute_dtype
        self.cross_entropy_reduction = cfg["cross_entropy_reduction"]
        self.alignment_mode = alignment_mode

        self.token_embedding = nn.Embedding(v, h)
        self.position_embedding = nn.Parameter(torch.zeros(s, h, dtype=torch.float32))
        self.ln1 = nn.LayerNorm(h, eps=cfg["layer_norm_eps"])
        self.q_proj = nn.Linear(h, h)
        self.k_proj = nn.Linear(h, h)
        self.v_proj = nn.Linear(h, h)
        self.o_proj = nn.Linear(h, h)
        self.ln2 = nn.LayerNorm(h, eps=cfg["layer_norm_eps"])
        self.fc1 = nn.Linear(h, f)
        self.fc2 = nn.Linear(f, h)
        self.lm_head = nn.Linear(h, v)

        self.load_shared_weights(weights)

    def load_shared_weights(self, weights):
        self.token_embedding.weight.data.copy_(torch.from_numpy(weights["token_embedding"]))
        self.position_embedding.data.copy_(torch.from_numpy(weights["position_embedding"]))
        self.ln1.weight.data.copy_(torch.from_numpy(weights["ln1_weight"]))
        self.ln1.bias.data.copy_(torch.from_numpy(weights["ln1_bias"]))
        self.q_proj.weight.data.copy_(torch.from_numpy(weights["q_proj_weight"]))
        self.q_proj.bias.data.copy_(torch.from_numpy(weights["q_proj_bias"]))
        self.k_proj.weight.data.copy_(torch.from_numpy(weights["k_proj_weight"]))
        self.k_proj.bias.data.copy_(torch.from_numpy(weights["k_proj_bias"]))
        self.v_proj.weight.data.copy_(torch.from_numpy(weights["v_proj_weight"]))
        self.v_proj.bias.data.copy_(torch.from_numpy(weights["v_proj_bias"]))
        self.o_proj.weight.data.copy_(torch.from_numpy(weights["o_proj_weight"]))
        self.o_proj.bias.data.copy_(torch.from_numpy(weights["o_proj_bias"]))
        self.ln2.weight.data.copy_(torch.from_numpy(weights["ln2_weight"]))
        self.ln2.bias.data.copy_(torch.from_numpy(weights["ln2_bias"]))
        self.fc1.weight.data.copy_(torch.from_numpy(weights["fc1_weight"]))
        self.fc1.bias.data.copy_(torch.from_numpy(weights["fc1_bias"]))
        self.fc2.weight.data.copy_(torch.from_numpy(weights["fc2_weight"]))
        self.fc2.bias.data.copy_(torch.from_numpy(weights["fc2_bias"]))
        self.lm_head.weight.data.copy_(torch.from_numpy(weights["lm_head_weight"]))
        self.lm_head.bias.data.copy_(torch.from_numpy(weights["lm_head_bias"]))

    def forward(self, input_ids, labels, causal_mask):
        batch_size, seq_len = input_ids.shape
        pos = self.position_embedding[:seq_len].unsqueeze(0)
        x = self.token_embedding(input_ids).to(self.compute_dtype)
        x = x + pos.to(x.dtype)

        h = self.ln1(x.float()).to(x.dtype)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_scores = attn_scores + causal_mask.to(attn_scores.dtype)

        # Reference numerics: upcast before softmax, then cast back.
        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(q.dtype)

        attn_ctx = torch.matmul(attn_probs, v)
        attn_ctx = attn_ctx.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        x = x + self.o_proj(attn_ctx)

        m = self.ln2(x.float()).to(x.dtype)
        x = x + self.fc2(exact_gelu(self.fc1(m)))
        logits = self.lm_head(x.float())

        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            reduction=self.cross_entropy_reduction,
        )

        debug = {
            "attn_scores_mean": float(attn_scores.float().mean().item()),
            "attn_scores_std": float(attn_scores.float().std().item()),
            "attn_probs_entropy": float(
                softmax_entropy(attn_probs[0, 0].float().detach().cpu().numpy())
            ),
            "logits_mean": float(logits.float().mean().item()),
            "logits_std": float(logits.float().std().item()),
        }
        alignment_tensors = None
        alignment_snapshot = None
        if self.alignment_mode:
            alignment_tensors = {
                "input_embeddings": to_numpy(self.token_embedding(input_ids)),
                "ln1_output": to_numpy(h),
                "q_proj_output": to_numpy(q),
                "k_proj_output": to_numpy(k),
                "v_proj_output": to_numpy(v),
                "attn_scores": to_numpy(attn_scores),
                "attn_probs": to_numpy(attn_probs),
                "attn_context": to_numpy(attn_ctx),
                "logits": to_numpy(logits),
            }
            alignment_snapshot = [
                summarize_named_array(name, array)
                for name, array in alignment_tensors.items()
            ]
        return loss, debug, alignment_snapshot, alignment_tensors


def named_parameter_summary(model):
    result = []
    for name, param in model.named_parameters():
        result.append(
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "requires_grad": bool(param.requires_grad),
            }
        )
    return result


def exact_gelu(x):
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return 0.5 * x * (1.0 + torch.erf(x * inv_sqrt2))


def to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def save_parameter_snapshot(model, output_dir):
    packed = {}
    index = []
    for i, (name, param) in enumerate(model.named_parameters()):
        key = f"param_{i:03d}"
        packed[key] = to_numpy(param)
        index.append(
            {
                "key": key,
                "name": name,
                "shape": list(param.shape),
            }
        )
    np.savez_compressed(os.path.join(output_dir, "alignment_parameters.npz"), **packed)
    with open(
        os.path.join(output_dir, "alignment_parameters_index.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(index, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/baseline")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--alignment-mode", action="store_true")
    parser.add_argument("--alignment-allow-update", action="store_true")
    parser.add_argument(
        "--no-alignment-diff-reduction",
        dest="alignment_diff_reduction",
        action="store_false",
        help="Keep deterministic alignment checks, but do not force extra diff-reduction steps such as compute_dtype=float32.",
    )
    parser.set_defaults(alignment_diff_reduction=True)
    args = parser.parse_args()

    cfg = get_case_config()
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.alignment_mode and args.alignment_diff_reduction:
        cfg["compute_dtype"] = "float32"
        if args.steps is None:
            cfg["steps"] = 1
    elif args.alignment_mode and args.steps is None:
        cfg["steps"] = 1

    determinism_info = None
    if args.alignment_mode:
        determinism_info = enable_alignment_determinism(
            int(cfg["seed"]),
            use_torch=True,
            use_torch_npu=True,
        )
    else:
        set_global_seed(int(cfg["seed"]))
        torch.manual_seed(int(cfg["seed"]))
        if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed_all"):
            torch.npu.manual_seed_all(int(cfg["seed"]))
    device = resolve_device()
    compute_dtype = to_torch_dtype(cfg["compute_dtype"])
    logger = JsonlLogger(args.output_dir, "baseline_torch_npu_train.py")

    logger.log("config", config=cfg)
    logger.log(
        "environment",
        framework="pytorch",
        device=str(device),
        torch_version=torch.__version__,
        torch_npu_available=True,
        alignment_mode=bool(args.alignment_mode),
        alignment_allow_update=bool(args.alignment_allow_update),
        alignment_diff_reduction=bool(args.alignment_diff_reduction),
        deterministic_algorithms=bool(args.alignment_mode),
        determinism_info=determinism_info,
    )

    weights = build_shared_weights(cfg)
    batch = build_fixed_batch(cfg)
    mask = build_causal_mask(cfg)
    logger.log("shared_weights_summary", tensors=summarize_arrays(weights)[:4])
    logger.log(
        "batch_summary",
        input_shape=list(batch["input_ids"].shape),
        label_shape=list(batch["labels"].shape),
        first_row=batch["input_ids"][0, :8].tolist(),
    )

    model = TinyCausalLM(cfg, weights, compute_dtype, alignment_mode=args.alignment_mode).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=cfg["adam_eps"],
        weight_decay=cfg["weight_decay"],
    )

    input_ids = torch.from_numpy(batch["input_ids"]).to(device=device, dtype=torch.long)
    labels = torch.from_numpy(batch["labels"]).to(device=device, dtype=torch.long)
    causal_mask = torch.from_numpy(mask).to(device=device, dtype=compute_dtype)

    logger.log("model_summary", parameters=named_parameter_summary(model)[:12])
    logger.log("phase", name="train_start")

    losses = []
    for step in range(1, int(cfg["steps"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, debug, alignment_snapshot, alignment_tensors = model(input_ids, labels, causal_mask)
        grad_norm = None
        update_applied = False
        if not args.alignment_mode or args.alignment_allow_update:
            loss.backward()
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += float(param.grad.float().norm().item()) ** 2
            grad_norm = grad_norm ** 0.5
            optimizer.step()
            update_applied = True
        losses.append(float(loss.item()))
        if args.alignment_mode and alignment_snapshot is not None:
            alignment_record = {
                "framework": "pytorch",
                "alignment_mode": True,
                "step": step,
                "compute_dtype": cfg["compute_dtype"],
                "softmax_path": "fp32_then_cast_back",
                "alignment_allow_update": bool(args.alignment_allow_update),
                "alignment_diff_reduction": bool(args.alignment_diff_reduction),
                "tensors": alignment_snapshot,
            }
            with open(
                os.path.join(args.output_dir, "alignment_snapshot.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(alignment_record, f, indent=2)
            np.savez_compressed(
                os.path.join(args.output_dir, "alignment_tensors.npz"),
                **alignment_tensors,
            )
            logger.log("alignment_snapshot", step=step, tensors=alignment_snapshot)
        logger.log(
            "train_step",
            step=step,
            loss=float(loss.item()),
            grad_norm=grad_norm,
            update_applied=update_applied,
            debug=debug,
        )

    if args.alignment_mode:
        save_parameter_snapshot(model, args.output_dir)

    summary = {
        "framework": "pytorch",
        "device": str(device),
        "losses": losses,
        "step1_loss": losses[0],
        "final_loss": losses[-1],
        "alignment_mode": bool(args.alignment_mode),
        "alignment_allow_update": bool(args.alignment_allow_update),
        "alignment_diff_reduction": bool(args.alignment_diff_reduction),
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.log("phase", name="train_end", summary=summary)


if __name__ == "__main__":
    main()
