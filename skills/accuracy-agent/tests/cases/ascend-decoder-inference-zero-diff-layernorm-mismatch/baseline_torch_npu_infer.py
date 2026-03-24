import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch_npu  # noqa: F401

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


def resolve_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    raise RuntimeError(
        "This baseline script requires a PyTorch torch_npu runtime with an "
        "available NPU device. CPU fallback is intentionally disabled."
    )


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


class TinyDecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        ffn_size = int(cfg["ffn_size"])
        self.num_heads = int(cfg["num_heads"])
        self.head_dim = hidden_size // self.num_heads
        self.scale = attention_scale(cfg)

        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)

    def load_shared_weights(self, weights):
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

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        normed = self.ln1(hidden_states)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_scores = attn_scores + causal_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_context = torch.matmul(attn_probs, v)
        attn_context = (
            attn_context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        )

        hidden_states = hidden_states + self.o_proj(attn_context)
        ffn_output = self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(hidden_states))))
        return hidden_states + ffn_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_case_config()
    logger = JsonlLogger(args.output_dir, "baseline_torch_npu_infer.py")
    determinism_info = enable_alignment_determinism(
        int(cfg["seed"]),
        use_torch=True,
        use_torch_npu=True,
    )
    device = resolve_device()

    logger.log(
        "environment",
        framework="pytorch",
        device=str(device),
        torch_version=torch.__version__,
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
    model = model.to(device)
    model.eval()

    output_records = []
    with torch.no_grad():
        for batch in batches:
            hidden_states = torch.from_numpy(batch["hidden_states"]).to(device=device, dtype=torch.float32)
            causal_mask = torch.from_numpy(build_causal_mask(hidden_states.shape[1])).to(
                device=device,
                dtype=torch.float32,
            )
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
