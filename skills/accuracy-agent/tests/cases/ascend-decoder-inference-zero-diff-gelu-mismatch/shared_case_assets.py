import json
import math
import os
import random
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np


CASE_CONFIG = {
    "seed": 20260324,
    "hidden_size": 64,
    "num_heads": 4,
    "ffn_size": 192,
    "max_seq_len": 12,
    "weight_scale": 0.02,
    "compute_dtype": "float32",
    "batch_specs": [
        {"name": "batch_00", "batch_size": 1, "seq_len": 5},
        {"name": "batch_01", "batch_size": 2, "seq_len": 7},
        {"name": "batch_02", "batch_size": 3, "seq_len": 11},
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlLogger:
    def __init__(self, output_dir: str, script_name: str):
        self.output_dir = output_dir
        self.script_name = script_name
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "run.log")

    def log(self, event: str, **payload) -> None:
        record = {"ts": utc_now(), "script": self.script_name, "event": event}
        record.update(payload)
        line = json.dumps(record, ensure_ascii=True, sort_keys=True)
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def enable_alignment_determinism(
    seed: int,
    use_torch: bool = False,
    use_torch_npu: bool = False,
    use_mindspore: bool = False,
) -> Dict[str, object]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["HCCL_DETERMINISTIC"] = "true"
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    set_global_seed(seed)

    result: Dict[str, object] = {
        "seed": seed,
        "env": {
            "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
            "HCCL_DETERMINISTIC": os.environ["HCCL_DETERMINISTIC"],
            "ASCEND_LAUNCH_BLOCKING": os.environ["ASCEND_LAUNCH_BLOCKING"],
        },
    }

    if use_torch:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        result["torch"] = {
            "manual_seed": seed,
            "use_deterministic_algorithms": True,
        }

    if use_torch_npu:
        try:
            import torch_npu

            torch_npu.npu.manual_seed_all(seed)
            torch_npu.npu.manual_seed(seed)
            result["torch_npu"] = {
                "manual_seed_all": seed,
                "manual_seed": seed,
            }
        except Exception as exc:
            result["torch_npu"] = {"status": f"unavailable: {exc!r}"}

    if use_mindspore:
        import mindspore as ms

        ms.set_seed(seed)
        ms.set_deterministic(True)
        result["mindspore"] = {
            "set_seed": seed,
            "set_deterministic": True,
        }

    return result


def get_case_config() -> Dict[str, object]:
    return dict(CASE_CONFIG)


def build_fixed_batches(cfg: Dict[str, object]) -> List[Dict[str, object]]:
    rng = np.random.default_rng(int(cfg["seed"]) + 11)
    batches = []
    hidden_size = int(cfg["hidden_size"])
    for spec in cfg["batch_specs"]:
        hidden_states = rng.standard_normal(
            (int(spec["batch_size"]), int(spec["seq_len"]), hidden_size)
        ).astype(np.float32)
        batches.append(
            {
                "name": spec["name"],
                "hidden_states": hidden_states,
            }
        )
    return batches


def build_causal_mask(seq_len: int) -> np.ndarray:
    mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
    mask = mask * -10000.0
    return mask.reshape(1, 1, seq_len, seq_len)


def build_shared_weights(cfg: Dict[str, object]) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(cfg["seed"]) + 29)
    hidden_size = int(cfg["hidden_size"])
    ffn_size = int(cfg["ffn_size"])
    scale = float(cfg["weight_scale"])

    def normal(shape):
        return (rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float32)

    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    return {
        "ln1_weight": np.ones((hidden_size,), dtype=np.float32),
        "ln1_bias": zeros((hidden_size,)),
        "q_proj_weight": normal((hidden_size, hidden_size)),
        "q_proj_bias": zeros((hidden_size,)),
        "k_proj_weight": normal((hidden_size, hidden_size)),
        "k_proj_bias": zeros((hidden_size,)),
        "v_proj_weight": normal((hidden_size, hidden_size)),
        "v_proj_bias": zeros((hidden_size,)),
        "o_proj_weight": normal((hidden_size, hidden_size)),
        "o_proj_bias": zeros((hidden_size,)),
        "ln2_weight": np.ones((hidden_size,), dtype=np.float32),
        "ln2_bias": zeros((hidden_size,)),
        "fc1_weight": normal((ffn_size, hidden_size)),
        "fc1_bias": zeros((ffn_size,)),
        "fc2_weight": normal((hidden_size, ffn_size)),
        "fc2_bias": zeros((hidden_size,)),
    }


def summarize_named_array(name: str, array: np.ndarray, preview_items: int = 8) -> Dict[str, object]:
    flat = array.reshape(-1)
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "preview": flat[:preview_items].tolist(),
    }


def summarize_batch(batch: Dict[str, object]) -> Dict[str, object]:
    hidden_states = batch["hidden_states"]
    return {
        "name": batch["name"],
        "shape": list(hidden_states.shape),
        "dtype": str(hidden_states.dtype),
        "preview": hidden_states.reshape(-1)[:8].tolist(),
    }


def attention_scale(cfg: Dict[str, object]) -> float:
    head_dim = int(cfg["hidden_size"]) // int(cfg["num_heads"])
    return 1.0 / math.sqrt(head_dim)


def save_shared_inputs(output_dir: str, batches: List[Dict[str, object]]) -> None:
    packed = {}
    for batch in batches:
        packed[f"{batch['name']}_hidden_states"] = batch["hidden_states"].astype(np.float32)
    np.savez_compressed(os.path.join(output_dir, "shared_inputs.npz"), **packed)


def save_shared_weights(output_dir: str, weights: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(os.path.join(output_dir, "shared_weights.npz"), **weights)


def save_outputs(output_dir: str, outputs: List[Dict[str, object]]) -> None:
    packed = {}
    summary = []
    for item in outputs:
        name = item["name"]
        output = np.asarray(item["output"], dtype=np.float32)
        packed[f"{name}_output"] = output
        summary.append(
            {
                "name": name,
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "preview": output.reshape(-1)[:8].tolist(),
            }
        )

    np.savez_compressed(os.path.join(output_dir, "decoder_outputs.npz"), **packed)
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"outputs": summary}, handle, indent=2)
