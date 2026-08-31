#!/usr/bin/env python3
"""
MPI Benchmark Runner

Runs the MPI benchmark with 1, 2, 4, 8 processes and generates comparison tables.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def run_mpi_benchmark(num_procs: int) -> bool:
    """Run MPI benchmark with given number of processes."""
    print(f"\n{'='*80}")
    print(f"RUNNING MPI BENCHMARK WITH {num_procs} PROCESS(ES)")
    print(f"{'='*80}\n")

    script_path = Path(__file__).parent.parent / "spell_checker" / "mpi_benchmark.py"

    # Try different MPI executables
    mpi_commands = ["mpiexec", "mpirun"]

    for mpi_cmd in mpi_commands:
        try:
            cmd = [mpi_cmd, "-n", str(num_procs), sys.executable, str(script_path)]
            print(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent),
                timeout=600,  # 10 minute timeout
                capture_output=False
            )

            if result.returncode == 0:
                return True
            else:
                print(f"Warning: {mpi_cmd} returned code {result.returncode}")

        except FileNotFoundError:
            print(f"  {mpi_cmd} not found, trying next...")
            continue
        except subprocess.TimeoutExpired:
            print(f"  Timeout after 600 seconds")
            return False
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"ERROR: Could not run MPI with {num_procs} processes")
    return False


def print_comparison_table():
    """Print comparison table from collected results."""
    base = Path(__file__).parent.parent
    mpi_file = base / "results" / "mpi_benchmark.json"
    seq_file = base / "results" / "sequential_baseline_fair.json"

    if not mpi_file.exists():
        print("ERROR: MPI results not found")
        return

    with open(mpi_file, 'r', encoding='utf-8') as f:
        mpi_data = json.load(f)

    # Load sequential baseline for comparison
    seq_times = {}
    if seq_file.exists():
        with open(seq_file, 'r', encoding='utf-8') as f:
            seq_data = json.load(f)
        for r in seq_data.get("results", []):
            key = (r["algorithm"], r["language"])
            seq_times[key] = r["ms_per_word"]

    # Organize results by algorithm, language, num_procs
    results_table = {}
    for run in mpi_data.get("runs", []):
        num_procs = run["num_procs"]
        for r in run["results"]:
            key = (r["algorithm"], r["language"])
            if key not in results_table:
                results_table[key] = {}
            results_table[key][num_procs] = r["ms_per_word"]

    # Print timing table
    print("\n" + "=" * 90)
    print("MPI SPEEDUP RESULTS (50K dict, 50 words)")
    print("=" * 90)

    proc_counts = sorted(set(run["num_procs"] for run in mpi_data.get("runs", [])))
    header = f"{'Algorithm':<22} | {'Lang':<4}"
    for p in proc_counts:
        header += f" | {p}-proc"
    print(header)
    print("-" * 90)

    for (algo, lang), times in sorted(results_table.items()):
        row = f"{algo:<22} | {lang:<4}"
        for p in proc_counts:
            if p in times:
                row += f" | {times[p]:>6.1f}ms"
            else:
                row += f" | {'N/A':>7}"
        print(row)

    print("-" * 90)

    # Print speedup table
    if seq_times and len(proc_counts) > 1:
        print("\n" + "=" * 90)
        print("SPEEDUP TABLE (vs 1-process baseline)")
        print("=" * 90)

        header = f"{'Algorithm':<22} | {'Lang':<4}"
        for p in proc_counts[1:]:  # Skip 1-proc
            header += f" | {p}x speedup"
        print(header)
        print("-" * 90)

        for (algo, lang), times in sorted(results_table.items()):
            row = f"{algo:<22} | {lang:<4}"

            # Use 1-proc MPI time as baseline, or sequential if not available
            baseline = times.get(1, seq_times.get((algo, lang), 0))

            for p in proc_counts[1:]:
                if p in times and baseline > 0:
                    speedup = baseline / times[p]
                    efficiency = speedup / p * 100
                    row += f" | {speedup:>5.2f}x ({efficiency:>4.0f}%)"
                else:
                    row += f" | {'N/A':>12}"
            print(row)

        print("-" * 90)

    # Print efficiency analysis
    if len(proc_counts) > 1:
        print("\n" + "=" * 60)
        print("PARALLEL EFFICIENCY ANALYSIS")
        print("=" * 60)

        for (algo, lang), times in sorted(results_table.items()):
            baseline = times.get(1, seq_times.get((algo, lang), 0))
            if baseline <= 0:
                continue

            print(f"\n{algo} + {lang}:")
            for p in proc_counts:
                if p in times:
                    if p == 1:
                        print(f"  {p} proc:  {times[p]:>7.1f} ms/word (baseline)")
                    else:
                        speedup = baseline / times[p]
                        efficiency = speedup / p * 100
                        ideal = baseline / p
                        overhead = times[p] - ideal
                        print(f"  {p} procs: {times[p]:>7.1f} ms/word | "
                              f"speedup: {speedup:.2f}x | "
                              f"efficiency: {efficiency:.0f}% | "
                              f"overhead: {overhead:.1f}ms")


def main():
    print("=" * 80)
    print("MPI BENCHMARK RUNNER")
    print("=" * 80)

    # Check if mpi4py is installed
    try:
        import mpi4py
        print(f"mpi4py version: {mpi4py.__version__}")
    except ImportError:
        print("ERROR: mpi4py not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mpi4py"])

    # Run configurations
    process_counts = [1, 2, 4, 8]
    successful = []

    for num_procs in process_counts:
        if run_mpi_benchmark(num_procs):
            successful.append(num_procs)
            print(f"\n[OK] {num_procs}-process run completed")
        else:
            print(f"\n[FAIL] {num_procs}-process run failed")

    # Print comparison
    if successful:
        print("\n" + "=" * 80)
        print("ALL RUNS COMPLETED")
        print("=" * 80)
        print(f"Successful runs: {successful}")

        print_comparison_table()

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
