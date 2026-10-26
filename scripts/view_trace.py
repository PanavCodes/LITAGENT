"""Pretty-printer CLI for LitAgent JSONL trace logs.

Per PRD.md §7.7 (FR-OBS-3):
Renders node execution steps, timings, inputs, outputs, and errors in a clear terminal layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pretty_print_trace(trace_path: Path) -> None:
    if not trace_path.exists():
        print(f"Error: Trace file not found: {trace_path}")
        return

    print("=" * 80)
    print(f"LITAGENT EXECUTION TRACE: {trace_path.name}")
    print("=" * 80)

    total_duration = 0.0
    step_num = 1

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            node = ev.get("node", "unknown").upper()
            duration = ev.get("duration", 0.0)
            total_duration += duration
            err = ev.get("error")

            status = f"[FAILED: {err}]" if err else "[OK]"
            print(f"\nStep {step_num}: {node:<18} {status:<12} (Duration: {duration*1000:.1f}ms)")
            print(f"  Input : {ev.get('input_summary', '')}")
            print(f"  Output: {ev.get('output_summary', '')}")
            step_num += 1

    print("-" * 80)
    print(f"Summary: {step_num - 1} steps completed in {total_duration*1000:.1f}ms total trace time.")
    print("=" * 80 + "\n")


def list_traces(traces_dir: Path) -> None:
    if not traces_dir.exists():
        print("No traces found.")
        return
    files = sorted(traces_dir.glob("*.jsonl"), reverse=True)
    if not files:
        print("No trace files found in results/traces/.")
        return
    print(f"Available traces in {traces_dir}:")
    for idx, f in enumerate(files[:10], 1):
        print(f"  [{idx}] {f.name} ({f.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="LitAgent Trace Viewer")
    parser.add_argument("file", nargs="?", default=None, help="Path to trace JSONL file")
    parser.add_argument("--list", action="store_true", help="List recent traces")
    args = parser.parse_args()

    traces_dir = Path("results/traces")
    if args.list or args.file is None:
        list_traces(traces_dir)
        if not args.file and not args.list:
            files = sorted(traces_dir.glob("*.jsonl"), reverse=True)
            if files:
                print(f"\nDisplaying most recent trace: {files[0].name}")
                pretty_print_trace(files[0])
    else:
        pretty_print_trace(Path(args.file))


if __name__ == "__main__":
    main()
