import argparse
import json
import math
import os

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor, nn
import mindspore.mint as mint
import mindspore.mint.nn as mint_nn
import mindspore.mint.nn.functional as F

from shared_case_assets import (
    JsonlLogger,
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


def resolve_device_target():
    try:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        return "Ascend"
    except Exception as exc:
        raise RuntimeError(
            "This target script requires a MindSpore Ascend runtime. "
            "CPU fallback is intentionally disabled for this migration case."
        ) from exc


def to_ms_dtype(name: str):
    return {
        "float16": ms.float16,
        "float32": ms.float32,
        "bfloat16": ms.bfloat16,
    }[name]


class TinyCausalLM(nn.Cell):
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

        self.token_embedding = mint_nn.Embedding(v, h, dtype=compute_dtype)
        self.position_embedding = Parameter(Tensor(weights["position_embedding"], compute_dtype))
        self.ln1 = mint_nn.LayerNorm(h, eps=cfg["layer_norm_eps"], dtype=compute_dtype)
        self.q_proj = mint_nn.Linear(h, h, dtype=compute_dtype)
        self.k_proj = mint_nn.Linear(h, h, dtype=compute_dtype)
        self.v_proj = mint_nn.Linear(h, h, dtype=compute_dtype)
        self.o_proj = mint_nn.Linear(h, h, dtype=compute_dtype)
        self.ln2 = mint_nn.LayerNorm(h, eps=cfg["layer_norm_eps"], dtype=compute_dtype)
        self.fc1 = mint_nn.Linear(h, f, dtype=compute_dtype)
        self.fc2 = mint_nn.Linear(f, h, dtype=compute_dtype)
        self.lm_head = mint_nn.Linear(h, v, dtype=compute_dtype)

        self.load_shared_weights(weights)

    def load_shared_weights(self, weights):
        self.token_embedding.weight.set_data(Tensor(weights["token_embedding"], self.compute_dtype))
        self.position_embedding.set_data(Tensor(weights["position_embedding"], self.compute_dtype))
        self.ln1.weight.set_data(Tensor(weights["ln1_weight"], self.compute_dtype))
        self.ln1.bias.set_data(Tensor(weights["ln1_bias"], self.compute_dtype))
        self.q_proj.weight.set_data(Tensor(weights["q_proj_weight"], self.compute_dtype))
        self.q_proj.bias.set_data(Tensor(weights["q_proj_bias"], self.compute_dtype))
        self.k_proj.weight.set_data(Tensor(weights["k_proj_weight"], self.compute_dtype))
        self.k_proj.bias.set_data(Tensor(weights["k_proj_bias"], self.compute_dtype))
        self.v_proj.weight.set_data(Tensor(weights["v_proj_weight"], self.compute_dtype))
        self.v_proj.bias.set_data(Tensor(weights["v_proj_bias"], self.compute_dtype))
        self.o_proj.weight.set_data(Tensor(weights["o_proj_weight"], self.compute_dtype))
        self.o_proj.bias.set_data(Tensor(weights["o_proj_bias"], self.compute_dtype))
        self.ln2.weight.set_data(Tensor(weights["ln2_weight"], self.compute_dtype))
        self.ln2.bias.set_data(Tensor(weights["ln2_bias"], self.compute_dtype))
        self.fc1.weight.set_data(Tensor(weights["fc1_weight"], self.compute_dtype))
        self.fc1.bias.set_data(Tensor(weights["fc1_bias"], self.compute_dtype))
        self.fc2.weight.set_data(Tensor(weights["fc2_weight"], self.compute_dtype))
        self.fc2.bias.set_data(Tensor(weights["fc2_bias"], self.compute_dtype))
        self.lm_head.weight.set_data(Tensor(weights["lm_head_weight"], self.compute_dtype))
        self.lm_head.bias.set_data(Tensor(weights["lm_head_bias"], self.compute_dtype))

    def construct(self, input_ids, labels, causal_mask):
        batch_size, seq_len = input_ids.shape
        positions = mint.arange(seq_len, dtype=ms.int32)
        pos = self.position_embedding[positions]

        x = self.token_embedding(input_ids)
        x = x + pos

        h = self.ln1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = mint.permute(mint.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = mint.permute(mint.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = mint.permute(mint.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        attn_scores = mint.matmul(q, mint.swapaxes(k, -1, -2)) * self.scale
        attn_scores = attn_scores + causal_mask.astype(attn_scores.dtype)

        if self.alignment_mode:
            attn_probs = mint.softmax(attn_scores.astype(ms.float32), dim=-1).astype(q.dtype)
        else:
            # Keep softmax in compute dtype for the current implementation.
            attn_probs = mint.softmax(attn_scores, dim=-1)

        attn_ctx = mint.matmul(attn_probs, v)
        attn_ctx = mint.reshape(mint.permute(attn_ctx, (0, 2, 1, 3)), (batch_size, seq_len, -1))
        x = x + self.o_proj(attn_ctx)

        m = self.ln2(x)
        x = x + self.fc2(exact_gelu(self.fc1(m)))
        logits = self.lm_head(x)

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss = F.cross_entropy(
            mint.reshape(shift_logits, (-1, shift_logits.shape[-1])),
            mint.reshape(shift_labels, (-1,)),
            reduction=self.cross_entropy_reduction,
        )

        debug = {
            "attn_scores_mean": float(attn_scores.astype(ms.float32).mean().asnumpy()),
            "attn_scores_std": float(attn_scores.astype(ms.float32).std().asnumpy()),
            "attn_probs_entropy": float(
                softmax_entropy(attn_probs[0, 0].astype(ms.float32).asnumpy())
            ),
            "logits_mean": float(logits.astype(ms.float32).mean().asnumpy()),
            "logits_std": float(logits.astype(ms.float32).std().asnumpy()),
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


def model_summary(model):
    result = []
    for name, param in model.parameters_and_names():
        result.append(
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
            }
        )
    return result


def exact_gelu(x):
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return 0.5 * x * (1.0 + mint.erf(x * inv_sqrt2))


def to_numpy(tensor):
    return tensor.astype(ms.float32).asnumpy()


def save_parameter_snapshot(model, output_dir):
    packed = {}
    index = []
    for i, (name, param) in enumerate(model.parameters_and_names()):
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
    parser.add_argument("--output-dir", default="runs/target")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--alignment-mode", action="store_true")
    parser.add_argument("--alignment-allow-update", action="store_true")
    args = parser.parse_args()

    cfg = get_case_config()
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.alignment_mode and args.steps is None:
        cfg["steps"] = 1

    determinism_info = enable_alignment_determinism(
        int(cfg["seed"]),
        use_mindspore=True,
    )
    device_target = resolve_device_target()
    compute_dtype = to_ms_dtype(cfg["compute_dtype"])
    logger = JsonlLogger(args.output_dir, "target_mindspore_train.py")

    logger.log("config", config=cfg)
    logger.log(
        "environment",
        framework="mindspore",
        device_target=device_target,
        mindspore_version=ms.__version__,
        mode="PYNATIVE_MODE",
        alignment_mode=bool(args.alignment_mode),
        alignment_allow_update=bool(args.alignment_allow_update),
        deterministic_algorithms=True,
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

    model = TinyCausalLM(cfg, weights, compute_dtype, alignment_mode=args.alignment_mode)
    optimizer = mint.optim.AdamW(
        model.trainable_params(),
        lr=cfg["learning_rate"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=cfg["adam_eps"],
        weight_decay=cfg["weight_decay"],
    )

    input_ids = Tensor(batch["input_ids"], ms.int32)
    labels = Tensor(batch["labels"], ms.int32)
    causal_mask = Tensor(mask, compute_dtype)

    logger.log("model_summary", parameters=model_summary(model)[:12])
    logger.log("phase", name="train_start")

    def forward_fn(batch_input_ids, batch_labels):
        return model(batch_input_ids, batch_labels, causal_mask)

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    losses = []
    for step in range(1, int(cfg["steps"]) + 1):
        grad_norm = None
        update_applied = False
        if args.alignment_mode and not args.alignment_allow_update:
            loss, debug, alignment_snapshot, alignment_tensors = forward_fn(input_ids, labels)
        else:
            (loss, debug, alignment_snapshot, alignment_tensors), grads = grad_fn(input_ids, labels)
            optimizer(grads)
            grad_norm_sq = 0.0
            for grad in grads:
                grad_norm_sq += float((grad.astype(ms.float32) ** 2).sum().asnumpy())
            grad_norm = grad_norm_sq ** 0.5
            update_applied = True
        loss_value = float(loss.asnumpy())
        losses.append(loss_value)
        if args.alignment_mode and alignment_snapshot is not None:
            alignment_record = {
                "framework": "mindspore",
                "alignment_mode": True,
                "step": step,
                "compute_dtype": cfg["compute_dtype"],
                "softmax_path": "fp32_then_cast_back",
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
        "framework": "mindspore",
        "device_target": device_target,
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
