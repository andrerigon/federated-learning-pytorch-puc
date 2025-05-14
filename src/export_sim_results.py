#!/usr/bin/env python
"""
export_sim_results.py
─────────────────────
Walk all `runs_*` directories, read every `final_metrics.json`, and produce a
single CSV with simulation parameters + results.

Columns
-------
scenario            – folder name, e.g. runs_500x500_s10_sr0.9
grid_size           – 500
sensor_count        – 10
success_rate        – 0.9
strategy            – FedAvg, AdaptiveAsync, …
run_timestamp       – from JSON (run_YYYYMMDD_hhmmss)
run_id              – directory name (run_…)
convergence_time    – minutes
total_updates       – communication rounds
final_accuracy
final_loss
peak_accuracy
total_messages
successful_updates
success_rate_comm
avg_model_size
avg_staleness
max_staleness
"""

from __future__ import annotations
import os, re, json, csv, argparse, sys
from typing import Tuple, Dict, Any, List
from pathlib import Path

# ───────── helper to parse folder name ─────────────────────
def _parse_scenario(name: str) -> Tuple[int, int, float]:
    g = re.search(r"(\d+)x", name)
    s = re.search(r"_s(\d+)_", name)
    r = re.search(r"sr([\d\.]+)", name)
    return (
        int(g.group(1)) if g else 0,
        int(s.group(1)) if s else 0,
        float(r.group(1)) if r else 0.0,
    )


def collect(runs_root: Path) -> List[Dict[str, Any]]:
    """Return a list of flattened records for every run found."""
    records: List[Dict[str, Any]] = []

    for scen_dir in runs_root.glob("runs_*"):
        if not scen_dir.is_dir():
            continue
        grid, sensors, sr = _parse_scenario(scen_dir.name)

        metrics_root = scen_dir / "aggregator_metrics"
        if not metrics_root.exists():
            continue

        for strategy_dir in metrics_root.iterdir():
            if not strategy_dir.is_dir():
                continue
            strategy = strategy_dir.name.replace("Strategy", "")

            for run_dir in strategy_dir.iterdir():
                if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                    continue
                meta_file = run_dir / "final_metrics.json"
                if not meta_file.exists():
                    continue

                try:
                    js = json.loads(meta_file.read_text())
                except Exception as err:
                    print(f"[WARN] Cannot parse {meta_file}: {err}", file=sys.stderr)
                    continue

                perf = js.get("performance", {})
                comm = js.get("communication", {})
                stale = js.get("staleness", {})

                records.append(
                    {
                        "scenario": scen_dir.name,
                        "grid_size": grid,
                        "sensor_count": sensors,
                        "success_rate": sr,
                        "strategy": strategy,
                        "run_timestamp": js.get("run_timestamp"),
                        "run_id": run_dir.name,
                        # core metrics -------------------------------------------------
                        "convergence_time": round((js.get("convergence_time") or 0) / 60, 3),
                        "total_updates": js.get("total_updates"),
                        # model performance
                        "final_accuracy": perf.get("final_accuracy"),
                        "peak_accuracy": perf.get("peak_accuracy"),
                        "final_loss": perf.get("final_loss"),
                        # communication
                        "total_messages": comm.get("total_messages"),
                        "successful_updates": comm.get("successful_updates"),
                        "success_rate_comm": comm.get("success_rate"),
                        "avg_model_size": comm.get("average_model_size"),
                        # staleness
                        "avg_staleness": stale.get("average"),
                        "max_staleness": stale.get("maximum"),
                    }
                )

    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Export all sim results to a CSV")
    ap.add_argument("--runs", default="./runs", help="Directory containing runs_* folders")
    ap.add_argument("--out", default="results_summary.csv", help="Output CSV file")
    args = ap.parse_args()

    runs_root = Path(args.runs).expanduser().resolve()
    if not runs_root.exists():
        print(f"[ERROR] {runs_root} does not exist", file=sys.stderr)
        sys.exit(1)

    recs = collect(runs_root)
    if not recs:
        print("[WARN] No results found – nothing written")
        sys.exit(0)

    # write CSV
    fieldnames = list(recs[0].keys())
    out_path = Path(args.out).expanduser().resolve()
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(recs)

    print(f"✅  Wrote {len(recs)} rows to {out_path}")


if __name__ == "__main__":
    main()