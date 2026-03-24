import argparse
import os

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn as mint_nn
import numpy as np
from mindspore import Tensor

from shared_case_assets import (
    JsonlLogger,
    attention_scale,
    build_causal_mask,
    build_fixed_batches,
    build_shared_weights,
    enable_alignment_determinism,
    get_case_config,
    save_outputs,
    save_shared_inputs,
    save_shared_weights,
    summarize_batch,
    summarize_named_array,
)


def resolve_device_target() -> str:
    try:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        return "Ascend"
    except Exception as exc:
        raise RuntimeError(
            "This target script requires a MindSpore Ascend runtime. "
            "CPU fallback is intentionally disabled."
        ) from exc


def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.astype(ms.float32).asnumpy()


def set_parameter_attr(cell, attr_candidates, array, label: str):
    tensor = Tensor(array, ms.float32)
    for attr_name in attr_candidates:
        if hasattr(cell, attr_name):
            getattr(cell, attr_name).set_data(tensor)
            return
    raise AttributeError(f"Unable to set {label}; none of {attr_candidates} exist on {type(cell)}")


def set_linear_params(layer, weight, bias):
    set_parameter_attr(layer, ("weight", "kernel"), weight, "linear weight")
    set_parameter_attr(layer, ("bias",), bias, "linear bias")


def set_layer_norm_params(layer, weight, bias):
    set_parameter_attr(layer, ("weight", "gamma"), weight, "layer norm weight")
    set_parameter_attr(layer, ("bias", "beta"), bias, "layer norm bias")


class TinyDecoderBlock(nn.Cell):
    def __init__(self, cfg):
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        ffn_size = int(cfg["ffn_size"])
        self.num_heads = int(cfg["num_heads"])
        self.head_dim = hidden_size // self.num_heads
        self.scale = attention_scale(cfg)

        self.ln1 = mint_nn.LayerNorm(hidden_size, eps=1e-5, dtype=ms.float32)
        self.q_proj = mint_nn.Linear(hidden_size, hidden_size, dtype=ms.float32)
        self.k_proj = mint_nn.Linear(hidden_size, hidden_size, dtype=ms.float32)
        self.v_proj = mint_nn.Linear(hidden_size, hidden_size, dtype=ms.float32)
        self.o_proj = mint_nn.Linear(hidden_size, hidden_size, dtype=ms.float32)
        self.ln2 = mint_nn.LayerNorm(hidden_size, eps=1e-5, dtype=ms.float32)
        self.fc1 = mint_nn.Linear(hidden_size, ffn_size, dtype=ms.float32)
        self.fc2 = mint_nn.Linear(ffn_size, hidden_size, dtype=ms.float32)
        self.gelu = mint_nn.GELU(approximate="none")

    def load_shared_weights(self, weights):
        set_layer_norm_params(self.ln1, weights["ln1_weight"], weights["ln1_bias"])
        set_linear_params(self.q_proj, weights["q_proj_weight"], weights["q_proj_bias"])
        set_linear_params(self.k_proj, weights["k_proj_weight"], weights["k_proj_bias"])
        set_linear_params(self.v_proj, weights["v_proj_weight"], weights["v_proj_bias"])
        set_linear_params(self.o_proj, weights["o_proj_weight"], weights["o_proj_bias"])
        set_layer_norm_params(self.ln2, weights["ln2_weight"], weights["ln2_bias"])
        set_linear_params(self.fc1, weights["fc1_weight"], weights["fc1_bias"])
        set_linear_params(self.fc2, weights["fc2_weight"], weights["fc2_bias"])

    def construct(self, hidden_states: Tensor, causal_mask: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        normed = self.ln1(hidden_states)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)

        q = mint.permute(mint.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = mint.permute(mint.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = mint.permute(mint.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        attn_scores = mint.matmul(q, mint.swapaxes(k, -1, -2)) * self.scale
        attn_scores = attn_scores + causal_mask
        attn_probs = mint.softmax(attn_scores, dim=-1)
        attn_context = mint.matmul(attn_probs, v)
        attn_context = mint.reshape(mint.permute(attn_context, (0, 2, 1, 3)), (batch_size, seq_len, -1))

        hidden_states = hidden_states + self.o_proj(attn_context)
        ffn_output = self.fc2(self.gelu(self.fc1(self.ln2(hidden_states))))
        return hidden_states + ffn_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/target")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_case_config()
    logger = JsonlLogger(args.output_dir, "target_mindspore_infer.py")
    determinism_info = enable_alignment_determinism(int(cfg["seed"]), use_mindspore=True)
    device_target = resolve_device_target()

    logger.log(
        "environment",
        framework="mindspore",
        device_target=device_target,
        mindspore_version=ms.__version__,
        mode="PYNATIVE_MODE",
        determinism_info=determinism_info,
    )
    logger.log("config", config=cfg)

    weights = build_shared_weights(cfg)
    batches = build_fixed_batches(cfg)
    save_shared_weights(args.output_dir, weights)
    save_shared_inputs(args.output_dir, batches)
    logger.log(
        "shared_weights",
        tensors=[summarize_named_array(name, array) for name, array in list(weights.items())[:4]],
    )
    logger.log("batches", batches=[summarize_batch(batch) for batch in batches])

    model = TinyDecoderBlock(cfg)
    model.load_shared_weights(weights)
    model.set_train(False)

    output_records = []
    for batch in batches:
        hidden_states = Tensor(batch["hidden_states"], ms.float32)
        causal_mask = Tensor(build_causal_mask(hidden_states.shape[1]), ms.float32)
        logger.log("inference_start", batch_name=batch["name"], shape=list(hidden_states.shape))
        output = model(hidden_states, causal_mask)
        output_np = to_numpy(output)
        output_records.append({"name": batch["name"], "output": output_np})
        logger.log(
            "inference_end",
            batch_name=batch["name"],
            output_summary=summarize_named_array(f"{batch['name']}_output", output_np),
        )

    save_outputs(args.output_dir, output_records)
    logger.log(
        "run_complete",
        output_file=os.path.join(args.output_dir, "decoder_outputs.npz"),
        batch_count=len(output_records),
    )


if __name__ == "__main__":
    main()
