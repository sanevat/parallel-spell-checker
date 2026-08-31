"""
Generate Speedup and Efficiency Tables for All Algorithms and Languages

Speedup S(p) = T(1) / T(p)
Efficiency E(p) = S(p) / p
"""

import json
from pathlib import Path

results_dir = Path(__file__).parent.parent.parent  # Go up to project root

# Word counts
word_counts = [250, 500, 1000, 1500, 2000, 2500, 3500, 5000]

# Define file paths for MK/EN data
file_info = [
    ('results/text_split/text_split_250.json', 250),
    ('results/text_split/text_split_500.json', 500),
    ('results/text_split/text_split_1K.json', 1000),
    ('results/text_split/text_split_1500.json', 1500),
    ('results/text_split/text_split_2K.json', 2000),
    ('results/text_split/text_split_2500.json', 2500),
    ('results/text_split/text_split_3500.json', 3500),
    ('results/text_split/text_split_5000.json', 5000),
]

# Algorithms
algorithms = ['Levenshtein', 'Damerau-Levenshtein', 'Myers Bit-Vector']
languages = ['MK', 'EN', 'TR']

# Data structure: {algorithm: {num_procs: {language: {word_count: time}}}}
data = {alg: {p: {lang: {} for lang in languages} for p in [1, 2, 4, 8]} for alg in algorithms}

# Load MK and EN data
for file_path, num_words in file_info:
    full_path = results_dir / file_path
    if not full_path.exists():
        continue

    with open(full_path, 'r') as f:
        results = json.load(f)

    text_split = results.get('text_split', {})

    for num_procs in ['1', '2', '4', '8']:
        if num_procs in text_split:
            for entry in text_split[num_procs]:
                alg = entry['algorithm']
                lang = entry['language']
                if alg in algorithms and lang in languages:
                    data[alg][int(num_procs)][lang][num_words] = entry['total_time_s']

# Load Turkish data
tr_files = {
    'Levenshtein': 'results/text_split/tr_levenshtein_best_results.json',
    'Damerau-Levenshtein': 'results/text_split/tr_damerau_best_results.json',
    'Myers Bit-Vector': 'results/text_split/tr_myers_best_results.json',
}

# Map word counts (handle variations like 2499 vs 2500)
word_count_map = {250: 250, 500: 500, 1000: 1000, 1500: 1500, 2000: 2000,
                  2499: 2500, 2500: 2500, 3500: 3500, 4998: 5000, 5000: 5000}

for alg, file_path in tr_files.items():
    full_path = results_dir / file_path
    if not full_path.exists():
        continue

    with open(full_path, 'r') as f:
        tr_results = json.load(f)

    for entry in tr_results.get('results', []):
        entry_alg = entry['algorithm']
        # Handle algorithm name variations
        if entry_alg == 'Myers-BitVector':
            entry_alg = 'Myers Bit-Vector'

        if entry_alg == alg:
            num_procs = entry['num_procs']
            misspelled = entry['misspelled']
            mapped_words = word_count_map.get(misspelled, misspelled)
            if mapped_words in word_counts:
                data[alg][num_procs]['TR'][mapped_words] = entry['total_time_s']

# Calculate speedup and efficiency
def calc_speedup(t1, tp):
    if t1 is None or tp is None or tp == 0:
        return None
    return t1 / tp

def calc_efficiency(speedup, p):
    if speedup is None:
        return None
    return speedup / p

# Print tables
print("=" * 120)
print("SPEEDUP AND EFFICIENCY TABLES")
print("=" * 120)
print()
print("Speedup S(p) = T(1) / T(p)")
print("Efficiency E(p) = S(p) / p")
print()

for alg in algorithms:
    print("=" * 120)
    print(f"ALGORITHM: {alg}")
    print("=" * 120)

    # Speedup table
    print("\n--- SPEEDUP ---")
    header = f"{'Words':<8}"
    for lang in languages:
        header += f" | {'S(2) ' + lang:>10} {'S(4) ' + lang:>10} {'S(8) ' + lang:>10}"
    print(header)
    print("-" * len(header))

    for wc in word_counts:
        row = f"{wc:<8}"
        for lang in languages:
            t1 = data[alg][1][lang].get(wc)
            for p in [2, 4, 8]:
                tp = data[alg][p][lang].get(wc)
                speedup = calc_speedup(t1, tp)
                if speedup is not None:
                    row += f" {speedup:>10.2f}"
                else:
                    row += f" {'N/A':>10}"
        print(row)

    # Efficiency table
    print("\n--- EFFICIENCY (%) ---")
    header = f"{'Words':<8}"
    for lang in languages:
        header += f" | {'E(2) ' + lang:>10} {'E(4) ' + lang:>10} {'E(8) ' + lang:>10}"
    print(header)
    print("-" * len(header))

    for wc in word_counts:
        row = f"{wc:<8}"
        for lang in languages:
            t1 = data[alg][1][lang].get(wc)
            for p in [2, 4, 8]:
                tp = data[alg][p][lang].get(wc)
                speedup = calc_speedup(t1, tp)
                efficiency = calc_efficiency(speedup, p)
                if efficiency is not None:
                    row += f" {efficiency*100:>9.1f}%"
                else:
                    row += f" {'N/A':>10}"
        print(row)

    print()

# Summary table - Average speedup and efficiency per algorithm and language
print("=" * 120)
print("SUMMARY: AVERAGE SPEEDUP AND EFFICIENCY")
print("=" * 120)
print()

print(f"{'Algorithm':<25} {'Lang':<5} | {'Avg S(2)':>10} {'Avg S(4)':>10} {'Avg S(8)':>10} | {'Avg E(2)':>10} {'Avg E(4)':>10} {'Avg E(8)':>10}")
print("-" * 105)

for alg in algorithms:
    for lang in languages:
        avg_speedups = {2: [], 4: [], 8: []}

        for wc in word_counts:
            t1 = data[alg][1][lang].get(wc)
            for p in [2, 4, 8]:
                tp = data[alg][p][lang].get(wc)
                speedup = calc_speedup(t1, tp)
                if speedup is not None:
                    avg_speedups[p].append(speedup)

        row = f"{alg:<25} {lang:<5} |"
        for p in [2, 4, 8]:
            if avg_speedups[p]:
                avg_s = sum(avg_speedups[p]) / len(avg_speedups[p])
                row += f" {avg_s:>10.2f}"
            else:
                row += f" {'N/A':>10}"
        row += " |"
        for p in [2, 4, 8]:
            if avg_speedups[p]:
                avg_s = sum(avg_speedups[p]) / len(avg_speedups[p])
                avg_e = avg_s / p * 100
                row += f" {avg_e:>9.1f}%"
            else:
                row += f" {'N/A':>10}"
        print(row)

print()
print("=" * 120)
