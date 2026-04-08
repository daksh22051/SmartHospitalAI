"""Verify required submission artifacts and proof checks.

Usage:
    python validation/verify_submission_artifacts.py \
        --report-path results/submission_artifact_verification.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Tuple


def _read_json(path: str) -> Tuple[bool, Dict[str, Any], str]:
    if not os.path.exists(path):
        return False, {}, f"missing file: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return False, {}, f"invalid json object in: {path}"
        return True, payload, "ok"
    except Exception as exc:
        return False, {}, f"failed to parse {path}: {exc}"


def _check_fields(payload: Dict[str, Any], required: List[str]) -> Tuple[bool, List[str]]:
    checks = payload.get("checks", {})
    if not isinstance(checks, dict):
        return False, ["checks object missing or invalid"]

    errors: List[str] = []
    for key in required:
        if checks.get(key) is not True:
            errors.append(f"checks.{key} != true")
    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify submission proof artifacts and required pass checks")
    parser.add_argument(
        "--report-path",
        default="results/submission_artifact_verification.json",
        help="Path to write verification report JSON",
    )
    args = parser.parse_args()

    required_specs = [
        (
            "constraint_proof_benchmark",
            os.path.join("results", "constraint_proof_benchmark.json"),
            ["runtime_within_limit", "all_runs_succeeded"],
        ),
        (
            "constraint_proof_docker",
            os.path.join("results", "constraint_proof_docker.json"),
            ["build_succeeded", "run_succeeded", "runtime_within_limit"],
        ),
        (
            "live_llm_baseline_proof",
            os.path.join("results", "live_llm_baseline_proof.json"),
            ["inference_succeeded", "policy_source_is_openai", "live_llm_requirement_pass"],
        ),
    ]

    report: Dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "overall_pass": True,
        "items": [],
    }

    for name, path, required_checks in required_specs:
        file_ok, payload, file_msg = _read_json(path)
        item: Dict[str, Any] = {
            "name": name,
            "path": path,
            "file_present": file_ok,
            "file_message": file_msg,
            "required_checks": required_checks,
            "checks_pass": False,
            "errors": [],
        }

        if file_ok:
            checks_ok, errors = _check_fields(payload, required_checks)
            item["checks_pass"] = checks_ok
            item["errors"] = errors
        else:
            item["errors"] = [file_msg]

        item["pass"] = bool(item["file_present"] and item["checks_pass"])
        if not item["pass"]:
            report["overall_pass"] = False

        report["items"].append(item)

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("VERIFICATION_RESULT=" + ("PASS" if report["overall_pass"] else "FAIL"))
    print("VERIFICATION_REPORT=" + args.report_path)

    if not report["overall_pass"]:
        for item in report["items"]:
            if item["pass"]:
                continue
            print(f"- {item['name']}: {', '.join(item['errors'])}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())