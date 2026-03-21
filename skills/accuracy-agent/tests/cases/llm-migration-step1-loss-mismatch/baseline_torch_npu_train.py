import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401

from shared_case_assets import (
    JsonlLogger,
    MODEL_PRESETS,
    apply_model_preset,
    attention_scale,
    build_causal_mask,
    build_fixed_batch,
    build_shared_weights,
    enable_alignment_determinism,
    get_case_config,
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


def exact_gelu(x: torch.Tensor) -> torch.Tensor:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return 0.5 * x * (1.0 + torch.erf(x * inv_sqrt2))


def tanh_gelu(x: torch.Tensor) -> torch.Tensor:
    coeff = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(coeff * (x + 0.044715 * x.pow(3))))


def apply_gelu(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "exact":
        return exact_gelu(x)
    if mode == "tanh":
        return tanh_gelu(x)
    raise ValueError(f"Unsupported gelu mode: {mode}")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


class TinyTransformerBlock(nn.Module):
    def __init__(self, cfg, compute_dtype):
        super().__init__()
        h = int(cfg["hidden_size"])
        f = int(cfg["ffn_size"])
        self.num_heads = int(cfg["num_heads"])
        self.head_dim = h // self.num_heads
        self.scale = attention_scale(cfg)

        self.ln1 = nn.LayerNorm(h, eps=float(cfg["layer_norm_eps"]))
        self.q_proj = nn.Linear(h, h)
        self.k_proj = nn.Linear(h, h)
        self.v_proj = nn.Linear(h, h)
        self.o_proj = nn.Linear(h, h)
        self.ln2 = nn.LayerNorm(h, eps=float(cfg["layer_norm_eps"]))
        self.fc1 = nn.Linear(h, f)
        self.fc2 = nn.Linear(f, h)
        self.to(dtype=compute_dtype)

    def load_shared_weights(self, weights, prefix: str):
        self.ln1.weight.data.copy_(torch.from_numpy(weights[f"{prefix}_ln1_weight"]))
        self.ln1.bias.data.copy_(torch.from_numpy(weights[f"{prefix}_ln1_bias"]))
        self.q_proj.weight.data.copy_(torch.from_numpy(weights[f"{prefix}_q_proj_weight"]))
        self.q_proj.bias.data.copy_(torch.from_numpy(weights[f"{prefix}_q_proj_bias"]))
        self.k_proj.weight.data.copy_(torch.from_numpy(weights[f"{prefix}_k_proj_weight"]))
        self.k_proj.bias.data.copy_(torch.from_numpy(weights[f"{prefix}_k_proj_bias"]))
        self.v_proj.weight.data.copy_(torch.from_numpy(weights[f"{prefix}_v_proj_weight"]))
        self.v_proj.bias.data.copy_(torch.from_numpy(weights[f"{prefix}_v_proj_bias"]))
        self.o_proj.weight.data.copy_(torch.from_numpy(weights[f"{prefix}_o_proj_weight"]))
        self.o_proj.bias.data.copy_(torch.from_numpy(weights[f"{prefix}_o_proj_bias"]))
        self.ln2.weight.data.copy_(torch.from_numpy(weights[f"{prefix}_ln2_weight"]))
        self.ln2.bias.data.copy_(torch.from_numpy(weights[f"{prefix}_ln2_bias"]))
        self.fc1.weight.data.copy_(torch.from_numpy(weights[f"{prefix}_fc1_weight"]))
        self.fc1.bias.data.copy_(torch.from_numpy(weights[f"{prefix}_fc1_bias"]))
        self.fc2.weight.data.copy_(torch.from_numpy(weights[f"{prefix}_fc2_weight"]))
        self.fc2.bias.data.copy_(torch.from_numpy(weights[f"{prefix}_fc2_bias"]))

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        softmax_mode: str,
        gelu_mode: str,
        capture_outputs: bool = False,
    ):
        batch_size, seq_len, _ = x.shape
        ln1_output = self.ln1(x)
        q = self.q_proj(ln1_output)
        k = self.k_proj(ln1_output)
        v = self.v_proj(ln1_output)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_scores = attn_scores + causal_mask.to(attn_scores.dtype)
        if softmax_mode == "fp32":
            attn_probs = F.softmax(attn_scores.float(), dim=-1).to(q.dtype)
        elif softmax_mode == "compute_dtype":
            attn_probs = F.softmax(attn_scores, dim=-1)
        else:
            raise ValueError(f"Unsupported softmax mode: {softmax_mode}")

        attn_context = torch.matmul(attn_probs, v)
        attn_context = (
            attn_context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        )
        x = x + self.o_proj(attn_context)
        ff = self.fc2(apply_gelu(self.fc1(self.ln2(x)), gelu_mode))
        x = x + ff

        block_outputs = None
        if capture_outputs:
            block_outputs = {
                "ln1_output": ln1_output,
                "q_proj_output": q,
                "k_proj_output": k,
                "v_proj_output": v,
                "attn_scores": attn_scores,
                "attn_probs": attn_probs,
                "attn_context": attn_context,
            }
        return x, block_outputs


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        cfg,
        weights,
        compute_dtype,
        softmax_mode: str,
        gelu_mode: str,
        alignment_mode: bool = False,
    ):
        super().__init__()
        h = int(cfg["hidden_size"])
        v = int(cfg["vocab_size"])
        s = int(cfg["seq_len"])
        self.compute_dtype = compute_dtype
        self.cross_entropy_reduction = cfg["cross_entropy_reduction"]
        self.softmax_mode = softmax_mode
        self.gelu_mode = gelu_mode
        self.alignment_mode = alignment_mode

        self.token_embedding = nn.Embedding(v, h)
        self.position_embedding = nn.Parameter(torch.zeros(s, h, dtype=compute_dtype))
        self.layers = nn.ModuleList(
            [TinyTransformerBlock(cfg, compute_dtype) for _ in range(int(cfg["num_layers"]))]
        )
        self.lm_head = nn.Linear(h, v)

        self.load_shared_weights(weights)
        self.to(dtype=compute_dtype)

    def load_shared_weights(self, weights):
        self.token_embedding.weight.data.copy_(torch.from_numpy(weights["token_embedding"]))
        self.position_embedding.data.copy_(
            torch.from_numpy(weights["position_embedding"]).to(self.position_embedding.dtype)
        )
        for layer_idx, layer in enumerate(self.layers):
            layer.load_shared_weights(weights, f"layer{layer_idx}")
        self.lm_head.weight.data.copy_(torch.from_numpy(weights["lm_head_weight"]))
        self.lm_head.bias.data.copy_(torch.from_numpy(weights["lm_head_bias"]))

    def forward(self, input_ids, labels, causal_mask):
        batch_size, seq_len = input_ids.shape
        pos = self.position_embedding[:seq_len].unsqueeze(0)
        x = self.token_embedding(input_ids).to(self.compute_dtype)
        x = x + pos.to(x.dtype)

        first_block_outputs = None
        for layer_idx, layer in enumerate(self.layers):
            capture_outputs = self.alignment_mode and layer_idx == 0
            x, block_outputs = layer(
                x,
                causal_mask,
                softmax_mode=self.softmax_mode,
                gelu_mode=self.gelu_mode,
                capture_outputs=capture_outputs,
            )
            if block_outputs is not None:
                first_block_outputs = block_outputs

        logits = self.lm_head(x)
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            reduction=self.cross_entropy_reduction,
        )

        if first_block_outputs is None:
            first_block_outputs = {}
        attn_scores = first_block_outputs.get("attn_scores")
        attn_probs = first_block_outputs.get("attn_probs")
        debug = {
            "softmax_mode": self.softmax_mode,
            "gelu_mode": self.gelu_mode,
            "attn_scores_mean": float(attn_scores.float().mean().item()) if attn_scores is not None else None,
            "attn_scores_std": float(attn_scores.float().std().item()) if attn_scores is not None else None,
            "attn_probs_entropy": (
                float(softmax_entropy(attn_probs[0, 0].float().detach().cpu().numpy()))
                if attn_probs is not None
                else None
            ),
            "logits_mean": float(logits.float().mean().item()),
            "logits_std": float(logits.float().std().item()),
        }

        alignment_snapshot = None
        alignment_tensors = None
        if self.alignment_mode and first_block_outputs is not None:
            alignment_tensors = {
                "input_embeddings": to_numpy(self.token_embedding(input_ids)),
                "ln1_output": to_numpy(first_block_outputs["ln1_output"]),
                "q_proj_output": to_numpy(first_block_outputs["q_proj_output"]),
                "k_proj_output": to_numpy(first_block_outputs["k_proj_output"]),
                "v_proj_output": to_numpy(first_block_outputs["v_proj_output"]),
                "attn_scores": to_numpy(first_block_outputs["attn_scores"]),
                "attn_probs": to_numpy(first_block_outputs["attn_probs"]),
                "attn_context": to_numpy(first_block_outputs["attn_context"]),
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


def save_parameter_snapshot(model, output_dir):
    packed = {}
    index = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        key = f"param_{idx:03d}"
        packed[key] = to_numpy(param)
        index.append({"key": key, "name": name, "shape": list(param.shape)})
    np.savez_compressed(os.path.join(output_dir, "alignment_parameters.npz"), **packed)
    with open(
        os.path.join(output_dir, "alignment_parameters_index.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(index, f, indent=2)


def build_runtime_config(args):
    cfg = get_case_config()
    apply_model_preset(cfg, args.model_preset)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.compute_dtype is not None:
        cfg["compute_dtype"] = args.compute_dtype
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.adam_eps is not None:
        cfg["adam_eps"] = args.adam_eps
    if args.layer_norm_eps is not None:
        cfg["layer_norm_eps"] = args.layer_norm_eps
    if args.alignment_mode and args.steps is None:
        cfg["steps"] = 1
    return cfg


def resolve_softmax_mode(args) -> str:
    return args.softmax_mode or "fp32"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/baseline")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="default")
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--adam-eps", type=float, default=None)
    parser.add_argument("--layer-norm-eps", type=float, default=None)
    parser.add_argument("--gelu-mode", choices=["exact", "tanh"], default="exact")
    parser.add_argument("--softmax-mode", choices=["compute_dtype", "fp32"], default=None)
    parser.add_argument("--alignment-mode", action="store_true")
    parser.add_argument("--alignment-allow-update", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_runtime_config(args)
    softmax_mode = resolve_softmax_mode(args)
    compute_dtype = to_torch_dtype(cfg["compute_dtype"])

    determinism_info = enable_alignment_determinism(
        int(cfg["seed"]),
        use_torch=True,
        use_torch_npu=True,
    )
    device = resolve_device()
    logger = JsonlLogger(args.output_dir, "baseline_torch_npu_train.py")

    logger.log("config", config=cfg)
    logger.log(
        "injection_switches",
        model_preset=args.model_preset,
        compute_dtype=cfg["compute_dtype"],
        softmax_mode=softmax_mode,
        gelu_mode=args.gelu_mode,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        adam_eps=cfg["adam_eps"],
        layer_norm_eps=cfg["layer_norm_eps"],
    )
    logger.log(
        "environment",
        framework="pytorch",
        device=str(device),
        torch_version=torch.__version__,
        torch_npu_available=True,
        alignment_mode=bool(args.alignment_mode),
        alignment_allow_update=bool(args.alignment_allow_update),
        deterministic_algorithms=True,
        determinism_info=determinism_info,
    )

    weights = build_shared_weights(cfg)
    batch = build_fixed_batch(cfg)
    mask = build_causal_mask(cfg)
    logger.log("shared_weights_summary", tensors=summarize_arrays(weights)[:6])
    logger.log(
        "batch_summary",
        input_shape=list(batch["input_ids"].shape),
        label_shape=list(batch["labels"].shape),
        first_row=batch["input_ids"][0, :8].tolist(),
    )

    model = TinyCausalLM(
        cfg,
        weights,
        compute_dtype=compute_dtype,
        softmax_mode=softmax_mode,
        gelu_mode=args.gelu_mode,
        alignment_mode=args.alignment_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=(float(cfg["adam_beta1"]), float(cfg["adam_beta2"])),
        eps=float(cfg["adam_eps"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    input_ids = torch.from_numpy(batch["input_ids"]).to(device=device, dtype=torch.long)
    labels = torch.from_numpy(batch["labels"]).to(device=device, dtype=torch.long)
    causal_mask = torch.from_numpy(mask).to(device=device, dtype=compute_dtype)

    logger.log("model_summary", parameters=named_parameter_summary(model)[:16])
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
            grad_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm_sq += float(param.grad.float().pow(2).sum().item())
            grad_norm = grad_norm_sq ** 0.5
            optimizer.step()
            update_applied = True
        loss_value = float(loss.item())
        losses.append(loss_value)

        if args.alignment_mode and alignment_snapshot is not None:
            alignment_record = {
                "framework": "pytorch",
                "alignment_mode": True,
                "step": step,
                "compute_dtype": cfg["compute_dtype"],
                "softmax_mode": softmax_mode,
                "gelu_mode": args.gelu_mode,
                "alignment_allow_update": bool(args.alignment_allow_update),
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
            loss=loss_value,
            grad_norm=grad_norm,
            update_applied=update_applied,
            debug=debug,
        )

    if args.alignment_mode:
        save_parameter_snapshot(model, args.output_dir)

    summary = {
        "framework": "pytorch",
        "device": str(device),
        "model_preset": args.model_preset,
        "compute_dtype": cfg["compute_dtype"],
        "softmax_mode": softmax_mode,
        "gelu_mode": args.gelu_mode,
        "weight_decay": cfg["weight_decay"],
        "adam_eps": cfg["adam_eps"],
        "layer_norm_eps": cfg["layer_norm_eps"],
        "losses": losses,
        "step1_loss": losses[0],
        "final_loss": losses[-1],
        "alignment_mode": bool(args.alignment_mode),
        "alignment_allow_update": bool(args.alignment_allow_update),
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.log("phase", name="train_end", summary=summary)


if __name__ == "__main__":
    main()
