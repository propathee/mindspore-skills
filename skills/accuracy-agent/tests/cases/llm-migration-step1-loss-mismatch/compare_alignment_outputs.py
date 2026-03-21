import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_ORDER = [
    "input_embeddings",
    "ln1_output",
    "q_proj_output",
    "k_proj_output",
    "v_proj_output",
    "attn_scores",
    "attn_probs",
    "attn_context",
    "logits",
]


def resolve_npz(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "alignment_tensors.npz")
        if os.path.exists(candidate):
            return candidate
    if path.endswith(".npz") and os.path.exists(path):
        return path
    raise FileNotFoundError(f"Could not find alignment_tensors.npz from: {path}")


def resolve_run_summary(path: str) -> str | None:
    if os.path.isdir(path):
        candidate = os.path.join(path, "run_summary.json")
        if os.path.exists(candidate):
            return candidate
    return None


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_arrays(
    left: np.ndarray,
    right: np.ndarray,
    warn_abs: float,
    fail_abs: float,
    warn_rel: float,
    fail_rel: float,
) -> Dict[str, object]:
    if left.shape != right.shape:
        return {
            "status": "shape_mismatch",
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }

    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    abs_diff = np.abs(left - right)
    denom = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-12)
    rel_diff = abs_diff / denom

    abs_flat_index = int(np.argmax(abs_diff))
    abs_max_index = list(np.unravel_index(abs_flat_index, abs_diff.shape))
    rel_flat_index = int(np.argmax(rel_diff))
    rel_max_index = list(np.unravel_index(rel_flat_index, rel_diff.shape))
    max_abs = float(abs_diff.reshape(-1)[abs_flat_index])
    max_rel = float(rel_diff.reshape(-1)[rel_flat_index])
    mean_abs = float(abs_diff.mean())
    mean_rel = float(rel_diff.mean())

    status = "pass"
    if max_abs > fail_abs or max_rel > fail_rel:
        status = "fail"
    elif max_abs > warn_abs or max_rel > warn_rel:
        status = "warn"

    return {
        "status": status,
        "shape": list(left.shape),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_rel_diff": max_rel,
        "mean_rel_diff": mean_rel,
        "max_abs_diff_index": abs_max_index,
        "left_value_at_max_abs": float(left[tuple(abs_max_index)]),
        "right_value_at_max_abs": float(right[tuple(abs_max_index)]),
        "max_rel_diff_index": rel_max_index,
        "left_value_at_max_rel": float(left[tuple(rel_max_index)]),
        "right_value_at_max_rel": float(right[tuple(rel_max_index)]),
    }


def summarize_report(
    tensor_reports: List[Dict[str, object]],
    loss_report: Dict[str, object] | None,
) -> Tuple[str, str | None]:
    first_issue = None
    overall = "pass"
    for report in tensor_reports:
        if report["status"] == "fail":
            overall = "fail"
            first_issue = report["name"]
            break
        if report["status"] == "warn" and overall == "pass":
            overall = "warn"
            first_issue = report["name"]
    if overall == "pass" and loss_report is not None and loss_report["status"] != "pass":
        overall = loss_report["status"]
        first_issue = "step1_loss"
    return overall, first_issue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline run dir or alignment_tensors.npz path")
    parser.add_argument("--target", required=True, help="Target run dir or alignment_tensors.npz path")
    parser.add_argument("--output", default=None, help="Optional path to write JSON report")
    parser.add_argument("--warn-abs", type=float, default=1e-6)
    parser.add_argument("--fail-abs", type=float, default=1e-5)
    parser.add_argument("--warn-rel", type=float, default=1e-6)
    parser.add_argument("--fail-rel", type=float, default=1e-4)
    args = parser.parse_args()

    baseline_npz = resolve_npz(args.baseline)
    target_npz = resolve_npz(args.target)
    baseline = load_npz(baseline_npz)
    target = load_npz(target_npz)
    missing_from_target = sorted(set(baseline) - set(target))
    missing_from_baseline = sorted(set(target) - set(baseline))

    tensor_reports = []
    all_names = []
    for name in DEFAULT_ORDER:
        if name in baseline and name in target:
            all_names.append(name)
    extra_names = sorted((set(baseline) & set(target)) - set(all_names))
    all_names.extend(extra_names)

    for name in all_names:
        report = compare_arrays(
            baseline[name],
            target[name],
            warn_abs=args.warn_abs,
            fail_abs=args.fail_abs,
            warn_rel=args.warn_rel,
            fail_rel=args.fail_rel,
        )
        report["name"] = name
        tensor_reports.append(report)

    loss_report = None
    baseline_summary_path = resolve_run_summary(args.baseline)
    target_summary_path = resolve_run_summary(args.target)
    if baseline_summary_path and target_summary_path:
        baseline_summary = load_json(baseline_summary_path)
        target_summary = load_json(target_summary_path)
        baseline_loss = float(baseline_summary["step1_loss"])
        target_loss = float(target_summary["step1_loss"])
        abs_diff = abs(baseline_loss - target_loss)
        rel_diff = abs_diff / max(abs(baseline_loss), abs(target_loss), 1e-12)
        status = "pass"
        if abs_diff > args.fail_abs or rel_diff > args.fail_rel:
            status = "fail"
        elif abs_diff > args.warn_abs or rel_diff > args.warn_rel:
            status = "warn"
        loss_report = {
            "status": status,
            "baseline_step1_loss": baseline_loss,
            "target_step1_loss": target_loss,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
        }

    overall_status, first_issue = summarize_report(tensor_reports, loss_report)
    report = {
        "baseline_npz": baseline_npz,
        "target_npz": target_npz,
        "thresholds": {
            "warn_abs": args.warn_abs,
            "fail_abs": args.fail_abs,
            "warn_rel": args.warn_rel,
            "fail_rel": args.fail_rel,
        },
        "overall_status": overall_status,
        "first_issue": first_issue,
        "missing_from_target": missing_from_target,
        "missing_from_baseline": missing_from_baseline,
        "tensor_reports": tensor_reports,
        "loss_report": loss_report,
    }

    if missing_from_target or missing_from_baseline:
        report["overall_status"] = "fail"
        if report["first_issue"] is None:
            report["first_issue"] = "missing_tensors"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0 if report["overall_status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
