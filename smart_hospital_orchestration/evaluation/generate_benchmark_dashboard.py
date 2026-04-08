"""Generate an HTML dashboard comparing random/heuristic/ppo benchmark runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(policy: str, data: Dict[str, Any]) -> str:
    s = data.get("summary", {})
    return (
        f"<tr><td>{policy}</td>"
        f"<td>{s.get('avg_reward_per_step', 0)}</td>"
        f"<td>{s.get('avg_final_waiting', 0)}</td>"
        f"<td>{s.get('avg_final_admitted', 0)}</td>"
        f"<td>{s.get('avg_final_critical_waiting', 0)}</td>"
        f"<td>{s.get('completion_rate', 0)}</td>"
        f"<td>{s.get('final_score', 0)}</td></tr>"
    )


def generate_dashboard(report_path: Path, output_path: Path) -> None:
    payload = _load_report(report_path)
    bench = payload.get("benchmark_comparison", {})

    rows = []
    for policy in ["random", "heuristic", "ppo"]:
        if policy in bench:
            rows.append(_row(policy, bench[policy]))

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Benchmark Comparison Dashboard</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #0f172a; }}
    h1 {{ margin-bottom: 6px; }}
    .meta {{ color: #475569; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 980px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 10px; text-align: left; }}
    th {{ background: #e2e8f0; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>Policy Benchmark Comparison</h1>
  <div class=\"meta\">Generated: {datetime.now(timezone.utc).isoformat()}</div>
  <table>
    <thead>
      <tr>
        <th>Policy</th>
        <th>Avg Reward/Step</th>
        <th>Avg Final Waiting</th>
        <th>Avg Final Admitted</th>
        <th>Avg Critical Waiting</th>
        <th>Completion Rate</th>
        <th>Final Score</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to grader report JSON")
    parser.add_argument(
        "--output",
        default="results/benchmark_dashboard.html",
        help="Output HTML path",
    )
    args = parser.parse_args()
    generate_dashboard(Path(args.report), Path(args.output))
    print(f"BENCHMARK_DASHBOARD={args.output}")


if __name__ == "__main__":
    main()
