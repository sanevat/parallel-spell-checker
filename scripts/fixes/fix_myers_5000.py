#!/usr/bin/env python3
"""
Re-run ONLY Myers Bit-Vector for 4 procs EN at 5000 words to fix anomaly.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent


def main():
    print("=" * 80)
    print("FIX: Re-running Myers Bit-Vector 4 procs for 5000 words")
    print("=" * 80)
    sys.stdout.flush()

    # Run MPI-4 for Myers only
    print("\nRunning MPI-4...")
    cmd = ["mpiexec", "-n", "4", "python", "-u",
           str(BASE / "scripts" / "fix_myers_5000_worker.py")]
    subprocess.run(cmd, cwd=str(BASE))

    # Load temp result
    temp_file = BASE / "results" / "fix_myers_5000_temp.json"
    if not temp_file.exists():
        print("ERROR: No temp file found")
        return

    with open(temp_file, 'r', encoding='utf-8') as f:
        new_data = json.load(f)

    # Load existing results
    results_file = BASE / "results" / "text_split_5000.json"
    with open(results_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    # Find and replace Myers Bit-Vector EN in 4 procs
    old_val = None
    new_val = new_data['results'][0]

    for i, entry in enumerate(all_results['text_split']['4']):
        if entry['algorithm'] == 'Myers Bit-Vector' and entry['language'] == 'EN':
            old_val = entry['total_time_s']
            all_results['text_split']['4'][i] = new_val
            break

    # Save updated results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated text_split_5000.json:")
    print(f"  Old value: {old_val:.1f}s")
    print(f"  New value: {new_val['total_time_s']:.1f}s")
    print("\nDone!")


if __name__ == "__main__":
    main()
