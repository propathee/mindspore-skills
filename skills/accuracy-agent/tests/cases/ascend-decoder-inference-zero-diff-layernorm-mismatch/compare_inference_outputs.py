import argparse
import json
import os
import sys
from typing import Dict

import numpy as np


def resolve_required_file(path: str, filename: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {filename} under: {path}")


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def compare_arrays(left: np.ndarray, right: np.ndarray) -> Dict[str, object]:
    if left.shape != right.shape:
        return {
            "status": "shape_mismatch",
            "left_shape": [int(dim) for dim in left.shape],
            "right_shape": [int(dim) for dim in right.shape],
        }

    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    diff = left - right
    abs_diff = np.abs(diff)
    denom = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-12)
    rel_diff = abs_diff / denom

    abs_max_index = [int(index) for index in np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)]
    rel_max_index = [int(index) for index in np.unravel_index(int(np.argmax(rel_diff)), rel_diff.shape)]
    nonzero_count = int(np.count_nonzero(abs_diff))
    max_abs = float(abs_diff[tuple(abs_max_index)])
    max_rel = float(rel_diff[tuple(rel_max_index)])

    status = "pass"
    if max_abs != 0.0 or max_rel != 0.0:
        status = "fail"

    return {
        "status": status,
        "shape": [int(dim) for dim in left.shape],
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(abs_diff.mean()),
        "max_rel_diff": max_rel,
        "mean_rel_diff": float(rel_diff.mean()),
        "l2_diff": float(np.linalg.norm(diff.reshape(-1), ord=2)),
        "nonzero_count": nonzero_count,
        "max_abs_diff_index": abs_max_index,
        "left_value_at_max_abs": float(left[tuple(abs_max_index)]),
        "right_value_at_max_abs": float(right[tuple(abs_max_index)]),
        "max_rel_diff_index": rel_max_index,
        "left_value_at_max_rel": float(left[tuple(rel_max_index)]),
        "right_value_at_max_rel": float(right[tuple(rel_max_index)]),
    }


def compare_bundle(name: str, baseline: Dict[str, np.ndarray], target: Dict[str, np.ndarray]):
    reports = []
    missing_from_target = sorted(set(baseline) - set(target))
    missing_from_baseline = sorted(set(target) - set(baseline))

    for key in sorted(set(baseline) & set(target)):
        report = compare_arrays(baseline[key], target[key])
        report["name"] = key
        reports.append(report)

    return {
        "bundle": name,
        "reports": reports,
        "missing_from_target": missing_from_target,
        "missing_from_baseline": missing_from_baseline,
    }


def print_bundle_summary(bundle_report):
    print(f"[{bundle_report['bundle']}]")
    if bundle_report["missing_from_target"] or bundle_report["missing_from_baseline"]:
        print(
            json.dumps(
                {
                    "missing_from_target": bundle_report["missing_from_target"],
                    "missing_from_baseline": bundle_report["missing_from_baseline"],
                },
                ensure_ascii=True,
            )
        )
    for report in bundle_report["reports"]:
        print(
            json.dumps(
                {
                    "name": report["name"],
                    "status": report["status"],
                    "shape": report["shape"],
                    "max_abs_diff": report["max_abs_diff"],
                    "mean_abs_diff": report["mean_abs_diff"],
                    "max_rel_diff": report["max_rel_diff"],
                    "mean_rel_diff": report["mean_rel_diff"],
                    "l2_diff": report["l2_diff"],
                    "nonzero_count": report["nonzero_count"],
                },
                ensure_ascii=True,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline run directory")
    parser.add_argument("--target", required=True, help="Target run directory")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    baseline_inputs = load_npz(resolve_required_file(args.baseline, "shared_inputs.npz"))
    target_inputs = load_npz(resolve_required_file(args.target, "shared_inputs.npz"))
    baseline_weights = load_npz(resolve_required_file(args.baseline, "shared_weights.npz"))
    target_weights = load_npz(resolve_required_file(args.target, "shared_weights.npz"))
    baseline_outputs = load_npz(resolve_required_file(args.baseline, "decoder_outputs.npz"))
    target_outputs = load_npz(resolve_required_file(args.target, "decoder_outputs.npz"))

    input_report = compare_bundle("shared_inputs", baseline_inputs, target_inputs)
    weight_report = compare_bundle("shared_weights", baseline_weights, target_weights)
    output_report = compare_bundle("decoder_outputs", baseline_outputs, target_outputs)

    print_bundle_summary(input_report)
    print_bundle_summary(weight_report)
    print_bundle_summary(output_report)

    overall_status = "pass"
    first_issue = None
    for bundle_report in [input_report, weight_report, output_report]:
        if bundle_report["missing_from_target"] or bundle_report["missing_from_baseline"]:
            overall_status = "fail"
            first_issue = first_issue or f"{bundle_report['bundle']}:missing"
        for report in bundle_report["reports"]:
            if report["status"] != "pass":
                overall_status = "fail"
                first_issue = first_issue or report["name"]
                break
        if overall_status == "fail" and first_issue is not None:
            break

    result = {
        "overall_status": overall_status,
        "first_issue": first_issue,
        "input_report": input_report,
        "weight_report": weight_report,
        "output_report": output_report,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    print("[overall]")
    print(json.dumps({"overall_status": overall_status, "first_issue": first_issue}, ensure_ascii=True))
    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
