import json
import math
import os
import random
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np


CASE_CONFIG = {
    "seed": 20260320,
    "vocab_size": 2048,
    "seq_len": 64,
    "batch_size": 4,
    "hidden_size": 128,
    "num_heads": 4,
    "num_layers": 1,
    "ffn_size": 256,
    "learning_rate": 5e-4,
    "weight_scale": 0.02,
    "steps": 12,
    "compute_dtype": "bfloat16",
    "label_shift": 1,
    "layer_norm_eps": 1e-5,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "weight_decay": 0.01,
    "cross_entropy_reduction": "mean",
}


MODEL_PRESETS = {
    "default": {},
    "large": {
        "seq_len": 1024,
        "batch_size": 2,
        "hidden_size": 512,
        "num_heads": 8,
        "num_layers": 2,
        "ffn_size": 2048,
    },
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
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


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
    os.environ["NCCL_DETERMINISTIC"] = "1"
    set_global_seed(seed)

    result: Dict[str, object] = {
        "seed": seed,
        "env": {
            "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
            "HCCL_DETERMINISTIC": os.environ["HCCL_DETERMINISTIC"],
            "ASCEND_LAUNCH_BLOCKING": os.environ["ASCEND_LAUNCH_BLOCKING"],
            "NCCL_DETERMINISTIC": os.environ["NCCL_DETERMINISTIC"],
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


def apply_model_preset(cfg: Dict[str, object], preset: str) -> Dict[str, object]:
    if preset not in MODEL_PRESETS:
        raise KeyError(f"Unknown model preset: {preset}")
    cfg.update(MODEL_PRESETS[preset])
    return cfg


def build_fixed_batch(cfg: Dict[str, object]) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(cfg["seed"]) + 11)
    input_ids = rng.integers(
        low=0,
        high=int(cfg["vocab_size"]),
        size=(int(cfg["batch_size"]), int(cfg["seq_len"])),
        dtype=np.int32,
    )
    labels = np.roll(input_ids, shift=-int(cfg["label_shift"]), axis=1)
    labels[:, -1] = 0
    return {"input_ids": input_ids, "labels": labels}


def build_causal_mask(cfg: Dict[str, object]) -> np.ndarray:
    seq_len = int(cfg["seq_len"])
    mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
    mask = mask * -10000.0
    return mask.reshape(1, 1, seq_len, seq_len)


def build_shared_weights(cfg: Dict[str, object]) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(cfg["seed"]) + 29)
    h = int(cfg["hidden_size"])
    f = int(cfg["ffn_size"])
    v = int(cfg["vocab_size"])
    s = int(cfg["seq_len"])
    scale = float(cfg["weight_scale"])

    def normal(shape):
        return (rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float32)

    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    weights = {
        "token_embedding": normal((v, h)),
        "position_embedding": normal((s, h)),
        "lm_head_weight": normal((v, h)),
        "lm_head_bias": zeros((v,)),
    }

    for layer_idx in range(int(cfg["num_layers"])):
        prefix = f"layer{layer_idx}"
        weights.update(
            {
                f"{prefix}_ln1_weight": np.ones((h,), dtype=np.float32),
                f"{prefix}_ln1_bias": zeros((h,)),
                f"{prefix}_q_proj_weight": normal((h, h)),
                f"{prefix}_q_proj_bias": zeros((h,)),
                f"{prefix}_k_proj_weight": normal((h, h)),
                f"{prefix}_k_proj_bias": zeros((h,)),
                f"{prefix}_v_proj_weight": normal((h, h)),
                f"{prefix}_v_proj_bias": zeros((h,)),
                f"{prefix}_o_proj_weight": normal((h, h)),
                f"{prefix}_o_proj_bias": zeros((h,)),
                f"{prefix}_ln2_weight": np.ones((h,), dtype=np.float32),
                f"{prefix}_ln2_bias": zeros((h,)),
                f"{prefix}_fc1_weight": normal((f, h)),
                f"{prefix}_fc1_bias": zeros((f,)),
                f"{prefix}_fc2_weight": normal((h, f)),
                f"{prefix}_fc2_bias": zeros((h,)),
            }
        )

    return weights


def summarize_arrays(arrays: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    summary = []
    for name, array in arrays.items():
        summary.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "mean": float(array.mean()),
                "std": float(array.std()),
            }
        )
    return summary


def summarize_named_array(
    name: str,
    array: np.ndarray,
    preview_items: int = 8,
) -> Dict[str, object]:
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


def softmax_entropy(probs: np.ndarray, eps: float = 1e-9) -> float:
    probs = np.clip(probs, eps, 1.0)
    entropy = -(probs * np.log(probs)).sum(axis=-1)
    return float(entropy.mean())


def head_dim(cfg: Dict[str, object]) -> int:
    return int(cfg["hidden_size"]) // int(cfg["num_heads"])


def attention_scale(cfg: Dict[str, object]) -> float:
    return 1.0 / math.sqrt(head_dim(cfg))
