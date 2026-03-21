import argparse
import json
import math
import os

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor, nn
import mindspore.mint as mint
import mindspore.mint.nn as mint_nn
import mindspore.mint.nn.functional as mint_F
import mindspore.ops as ops

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


def to_numpy(tensor):
    return tensor.astype(ms.float32).asnumpy()


def set_parameter_attr(cell, attr_candidates, array, dtype, label: str):
    tensor = Tensor(array, dtype)
    for attr_name in attr_candidates:
        if hasattr(cell, attr_name):
            getattr(cell, attr_name).set_data(tensor)
            return
    raise AttributeError(f"Unable to set {label}; none of {attr_candidates} exist on {type(cell)}")


def make_embedding(vocab_size: int, hidden_size: int, dtype, impl: str):
    if impl == "mint_nn":
        return mint_nn.Embedding(vocab_size, hidden_size, dtype=dtype)
    if impl == "nn":
        return nn.Embedding(vocab_size=vocab_size, embedding_size=hidden_size, dtype=dtype)
    raise ValueError(f"Unsupported embedding impl: {impl}")


def set_embedding_weight(layer, array, dtype):
    set_parameter_attr(layer, ("weight", "embedding_table"), array, dtype, "embedding weight")


def make_linear(in_channels: int, out_channels: int, dtype, impl: str):
    if impl == "mint_nn":
        return mint_nn.Linear(in_channels, out_channels, dtype=dtype)
    if impl == "nn":
        return nn.Dense(in_channels=in_channels, out_channels=out_channels, has_bias=True, dtype=dtype)
    raise ValueError(f"Unsupported linear impl: {impl}")


def set_linear_params(layer, weight, bias, dtype):
    set_parameter_attr(layer, ("weight", "kernel"), weight, dtype, "linear weight")
    set_parameter_attr(layer, ("bias",), bias, dtype, "linear bias")


def make_layer_norm(hidden_size: int, eps: float, dtype, impl: str):
    if impl == "mint_nn":
        return mint_nn.LayerNorm(hidden_size, eps=eps, dtype=dtype)
    if impl == "nn":
        return nn.LayerNorm((hidden_size,), epsilon=eps, dtype=dtype)
    raise ValueError(f"Unsupported layer norm impl: {impl}")


def set_layer_norm_params(layer, weight, bias, dtype):
    set_parameter_attr(layer, ("weight", "gamma"), weight, dtype, "layer norm weight")
    set_parameter_attr(layer, ("bias", "beta"), bias, dtype, "layer norm bias")


def make_gelu(impl: str):
    if impl == "mint_nn":
        return mint_nn.GELU(approximate="none")
    if impl == "nn":
        return nn.GELU()
    raise ValueError(f"Unsupported GELU impl: {impl}")


def do_matmul(left, right, impl: str):
    if impl == "mint":
        return mint.matmul(left, right)
    if impl == "ops":
        return ops.matmul(left, right)
    raise ValueError(f"Unsupported matmul impl: {impl}")


def do_softmax(values, impl: str, fp32_mode: bool):
    softmax_input = values.astype(ms.float32) if fp32_mode else values
    if impl == "mint":
        result = mint.softmax(softmax_input, dim=-1)
    elif impl == "ops":
        result = ops.softmax(softmax_input, axis=-1)
    else:
        raise ValueError(f"Unsupported softmax impl: {impl}")
    return result.astype(values.dtype) if fp32_mode else result


class TinyTransformerBlock(nn.Cell):
    def __init__(self, cfg, compute_dtype, linear_impl, layernorm_impl, gelu_impl, matmul_impl, softmax_impl):
        super().__init__()
        h = int(cfg["hidden_size"])
        f = int(cfg["ffn_size"])
        self.num_heads = int(cfg["num_heads"])
        self.head_dim = h // self.num_heads
        self.scale = attention_scale(cfg)
        self.compute_dtype = compute_dtype
        self.matmul_impl = matmul_impl
        self.softmax_impl = softmax_impl

        self.ln1 = make_layer_norm(h, float(cfg["layer_norm_eps"]), compute_dtype, layernorm_impl)
        self.q_proj = make_linear(h, h, compute_dtype, linear_impl)
        self.k_proj = make_linear(h, h, compute_dtype, linear_impl)
        self.v_proj = make_linear(h, h, compute_dtype, linear_impl)
        self.o_proj = make_linear(h, h, compute_dtype, linear_impl)
        self.ln2 = make_layer_norm(h, float(cfg["layer_norm_eps"]), compute_dtype, layernorm_impl)
        self.fc1 = make_linear(h, f, compute_dtype, linear_impl)
        self.fc2 = make_linear(f, h, compute_dtype, linear_impl)
        self.gelu = make_gelu(gelu_impl)

    def load_shared_weights(self, weights, prefix: str):
        set_layer_norm_params(self.ln1, weights[f"{prefix}_ln1_weight"], weights[f"{prefix}_ln1_bias"], self.compute_dtype)
        set_linear_params(self.q_proj, weights[f"{prefix}_q_proj_weight"], weights[f"{prefix}_q_proj_bias"], self.compute_dtype)
        set_linear_params(self.k_proj, weights[f"{prefix}_k_proj_weight"], weights[f"{prefix}_k_proj_bias"], self.compute_dtype)
        set_linear_params(self.v_proj, weights[f"{prefix}_v_proj_weight"], weights[f"{prefix}_v_proj_bias"], self.compute_dtype)
        set_linear_params(self.o_proj, weights[f"{prefix}_o_proj_weight"], weights[f"{prefix}_o_proj_bias"], self.compute_dtype)
        set_layer_norm_params(self.ln2, weights[f"{prefix}_ln2_weight"], weights[f"{prefix}_ln2_bias"], self.compute_dtype)
        set_linear_params(self.fc1, weights[f"{prefix}_fc1_weight"], weights[f"{prefix}_fc1_bias"], self.compute_dtype)
        set_linear_params(self.fc2, weights[f"{prefix}_fc2_weight"], weights[f"{prefix}_fc2_bias"], self.compute_dtype)

    def construct(self, x, causal_mask, softmax_mode: str, capture_outputs: bool = False):
        batch_size, seq_len, _ = x.shape
        ln1_output = self.ln1(x)
        q = self.q_proj(ln1_output)
        k = self.k_proj(ln1_output)
        v = self.v_proj(ln1_output)

        q = mint.permute(mint.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = mint.permute(mint.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = mint.permute(mint.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        attn_scores = do_matmul(q, mint.swapaxes(k, -1, -2), self.matmul_impl) * self.scale
        attn_scores = attn_scores + causal_mask.astype(attn_scores.dtype)
        attn_probs = do_softmax(
            attn_scores,
            impl=self.softmax_impl,
            fp32_mode=(softmax_mode == "fp32"),
        )

        attn_context = do_matmul(attn_probs, v, self.matmul_impl)
        attn_context = mint.reshape(mint.permute(attn_context, (0, 2, 1, 3)), (batch_size, seq_len, -1))
        x = x + self.o_proj(attn_context)
        ff = self.fc2(self.gelu(self.fc1(self.ln2(x))))
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


class TinyCausalLM(nn.Cell):
    def __init__(
        self,
        cfg,
        weights,
        compute_dtype,
        softmax_mode: str,
        alignment_mode: bool = False,
        embedding_impl: str = "mint_nn",
        linear_impl: str = "mint_nn",
        layernorm_impl: str = "mint_nn",
        gelu_impl: str = "mint_nn",
        matmul_impl: str = "mint",
        softmax_impl: str = "mint",
    ):
        super().__init__()
        h = int(cfg["hidden_size"])
        v = int(cfg["vocab_size"])
        s = int(cfg["seq_len"])
        self.compute_dtype = compute_dtype
        self.cross_entropy_reduction = cfg["cross_entropy_reduction"]
        self.softmax_mode = softmax_mode
        self.alignment_mode = alignment_mode
        self.embedding_impl = embedding_impl
        self.linear_impl = linear_impl
        self.layernorm_impl = layernorm_impl
        self.gelu_impl = gelu_impl
        self.matmul_impl = matmul_impl
        self.softmax_impl = softmax_impl

        self.token_embedding = make_embedding(v, h, compute_dtype, embedding_impl)
        self.position_embedding = Parameter(Tensor(np.zeros((s, h), dtype=np.float32), compute_dtype))
        self.layers = nn.CellList(
            [
                TinyTransformerBlock(
                    cfg,
                    compute_dtype=compute_dtype,
                    linear_impl=linear_impl,
                    layernorm_impl=layernorm_impl,
                    gelu_impl=gelu_impl,
                    matmul_impl=matmul_impl,
                    softmax_impl=softmax_impl,
                )
                for _ in range(int(cfg["num_layers"]))
            ]
        )
        self.lm_head = make_linear(h, v, compute_dtype, linear_impl)

        self.load_shared_weights(weights)

    def load_shared_weights(self, weights):
        set_embedding_weight(self.token_embedding, weights["token_embedding"], self.compute_dtype)
        self.position_embedding.set_data(Tensor(weights["position_embedding"], self.compute_dtype))
        for layer_idx, layer in enumerate(self.layers):
            layer.load_shared_weights(weights, f"layer{layer_idx}")
        set_linear_params(self.lm_head, weights["lm_head_weight"], weights["lm_head_bias"], self.compute_dtype)

    def construct(self, input_ids, labels, causal_mask):
        seq_len = input_ids.shape[1]
        positions = mint.arange(seq_len, dtype=ms.int32)
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[positions]

        first_block_outputs = None
        for layer_idx, layer in enumerate(self.layers):
            capture_outputs = self.alignment_mode and layer_idx == 0
            x, block_outputs = layer(x, causal_mask, softmax_mode=self.softmax_mode, capture_outputs=capture_outputs)
            if block_outputs is not None:
                first_block_outputs = block_outputs

        logits = self.lm_head(x)
        shift_logits = mint.reshape(logits[:, :-1, :], (-1, logits.shape[-1]))
        shift_labels = mint.reshape(labels[:, 1:], (-1,))
        loss = mint_F.cross_entropy(
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
            "embedding_impl": self.embedding_impl,
            "linear_impl": self.linear_impl,
            "layernorm_impl": self.layernorm_impl,
            "gelu_impl": self.gelu_impl,
            "matmul_impl": self.matmul_impl,
            "softmax_impl": self.softmax_impl,
            "attn_scores_mean": (
                float(attn_scores.astype(ms.float32).mean().asnumpy())
                if attn_scores is not None
                else None
            ),
            "attn_scores_std": (
                float(attn_scores.astype(ms.float32).std().asnumpy())
                if attn_scores is not None
                else None
            ),
            "attn_probs_entropy": (
                float(softmax_entropy(attn_probs[0, 0].astype(ms.float32).asnumpy()))
                if attn_probs is not None
                else None
            ),
            "logits_mean": float(logits.astype(ms.float32).mean().asnumpy()),
            "logits_std": float(logits.astype(ms.float32).std().asnumpy()),
        }

        alignment_snapshot = None
        alignment_tensors = None
        if self.alignment_mode and first_block_outputs:
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


def model_summary(model):
    result = []
    for name, param in model.parameters_and_names():
        result.append({"name": name, "shape": list(param.shape), "dtype": str(param.dtype)})
    return result


def save_parameter_snapshot(model, output_dir):
    packed = {}
    index = []
    for idx, (name, param) in enumerate(model.parameters_and_names()):
        key = f"param_{idx:03d}"
        packed[key] = to_numpy(param)
        index.append({"key": key, "name": name, "shape": list(param.shape)})
    np.savez_compressed(os.path.join(output_dir, "alignment_parameters.npz"), **packed)
    with open(os.path.join(output_dir, "alignment_parameters_index.json"), "w", encoding="utf-8") as f:
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
    parser.add_argument("--output-dir", default="runs/target")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="large")
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--adam-eps", type=float, default=None)
    parser.add_argument("--layer-norm-eps", type=float, default=None)
    parser.add_argument("--softmax-mode", choices=["compute_dtype", "fp32"], default=None)
    parser.add_argument("--embedding-impl", choices=["mint_nn", "nn"], default="mint_nn")
    parser.add_argument("--linear-impl", choices=["mint_nn", "nn"], default="mint_nn")
    parser.add_argument("--layernorm-impl", choices=["mint_nn", "nn"], default="mint_nn")
    parser.add_argument("--gelu-impl", choices=["mint_nn", "nn"], default="mint_nn")
    parser.add_argument("--matmul-impl", choices=["mint", "ops"], default="mint")
    parser.add_argument("--softmax-impl", choices=["mint", "ops"], default="mint")
    parser.add_argument("--optimizer-impl", choices=["nn", "mint"], default="nn")
    parser.add_argument("--alignment-mode", action="store_true")
    parser.add_argument("--alignment-allow-update", action="store_true")
    return parser.parse_args()


def create_optimizer(args, cfg, model):
    if args.optimizer_impl == "nn":
        return nn.Adam(
            model.trainable_params(),
            learning_rate=float(cfg["learning_rate"]),
            beta1=float(cfg["adam_beta1"]),
            beta2=float(cfg["adam_beta2"]),
            eps=float(cfg["adam_eps"]),
            weight_decay=float(cfg["weight_decay"]),
        )
    return mint.optim.AdamW(
        model.trainable_params(),
        lr=float(cfg["learning_rate"]),
        betas=(float(cfg["adam_beta1"]), float(cfg["adam_beta2"])),
        eps=float(cfg["adam_eps"]),
        weight_decay=float(cfg["weight_decay"]),
    )


def main():
    args = parse_args()
    cfg = build_runtime_config(args)
    softmax_mode = resolve_softmax_mode(args)
    compute_dtype = to_ms_dtype(cfg["compute_dtype"])

    determinism_info = enable_alignment_determinism(int(cfg["seed"]), use_mindspore=True)
    device_target = resolve_device_target()
    logger = JsonlLogger(args.output_dir, "target_mindspore_train.py")

    logger.log("config", config=cfg)
    logger.log(
        "injection_switches",
        model_preset=args.model_preset,
        compute_dtype=cfg["compute_dtype"],
        softmax_mode=softmax_mode,
        embedding_impl=args.embedding_impl,
        linear_impl=args.linear_impl,
        layernorm_impl=args.layernorm_impl,
        gelu_impl=args.gelu_impl,
        matmul_impl=args.matmul_impl,
        softmax_impl=args.softmax_impl,
        optimizer_impl=args.optimizer_impl,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        adam_eps=cfg["adam_eps"],
        layer_norm_eps=cfg["layer_norm_eps"],
    )
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
        alignment_mode=args.alignment_mode,
        embedding_impl=args.embedding_impl,
        linear_impl=args.linear_impl,
        layernorm_impl=args.layernorm_impl,
        gelu_impl=args.gelu_impl,
        matmul_impl=args.matmul_impl,
        softmax_impl=args.softmax_impl,
    )
    optimizer = create_optimizer(args, cfg, model)

    input_ids = Tensor(batch["input_ids"], ms.int32)
    labels = Tensor(batch["labels"], ms.int32)
    causal_mask = Tensor(mask, compute_dtype)

    logger.log("model_summary", parameters=model_summary(model)[:16])
    logger.log("phase", name="train_start")

    def forward_fn(batch_input_ids, batch_labels):
        return model(batch_input_ids, batch_labels, causal_mask)

    grad_fn = ms.value_and_grad(forward_fn, None, model.trainable_params(), has_aux=True)

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
                "softmax_mode": softmax_mode,
                "embedding_impl": args.embedding_impl,
                "linear_impl": args.linear_impl,
                "layernorm_impl": args.layernorm_impl,
                "gelu_impl": args.gelu_impl,
                "matmul_impl": args.matmul_impl,
                "softmax_impl": args.softmax_impl,
                "optimizer_impl": args.optimizer_impl,
                "alignment_allow_update": bool(args.alignment_allow_update),
                "tensors": alignment_snapshot,
            }
            with open(os.path.join(args.output_dir, "alignment_snapshot.json"), "w", encoding="utf-8") as f:
                json.dump(alignment_record, f, indent=2)
            np.savez_compressed(os.path.join(args.output_dir, "alignment_tensors.npz"), **alignment_tensors)
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
        "model_preset": args.model_preset,
        "compute_dtype": cfg["compute_dtype"],
        "softmax_mode": softmax_mode,
        "embedding_impl": args.embedding_impl,
        "linear_impl": args.linear_impl,
        "layernorm_impl": args.layernorm_impl,
        "gelu_impl": args.gelu_impl,
        "matmul_impl": args.matmul_impl,
        "softmax_impl": args.softmax_impl,
        "optimizer_impl": args.optimizer_impl,
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
