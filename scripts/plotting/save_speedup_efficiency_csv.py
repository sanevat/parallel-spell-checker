"""
Save Speedup and Efficiency Tables to CSV
"""

import json
import csv
from pathlib import Path

results_dir = Path(__file__).parent.parent.parent

# Word counts
word_counts = [250, 500, 1000, 1500, 2000, 2500, 3500, 5000]

# Define file paths
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

algorithms = ['Levenshtein', 'Damerau-Levenshtein', 'Myers Bit-Vector']
languages = ['MK', 'EN', 'TR']

# Data structure
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

word_count_map = {249: 250, 250: 250, 498: 500, 500: 500, 999: 1000, 1000: 1000,
                  1498: 1500, 1500: 1500, 1997: 2000, 2000: 2000,
                  2498: 2500, 2499: 2500, 2500: 2500, 3497: 3500, 3500: 3500,
                  4998: 5000, 5000: 5000}

for alg, file_path in tr_files.items():
    full_path = results_dir / file_path
    if not full_path.exists():
        continue
    with open(full_path, 'r') as f:
        tr_results = json.load(f)
    for entry in tr_results.get('results', []):
        entry_alg = entry['algorithm']
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

# Create output directory
output_dir = results_dir / 'results' / 'text_split'
output_dir.mkdir(parents=True, exist_ok=True)

# Save detailed CSV
csv_path = output_dir / 'speedup_efficiency_text_split.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Header
    writer.writerow(['Algorithm', 'Language', 'Words', 'T(1)', 'T(2)', 'T(4)', 'T(8)',
                     'S(2)', 'S(4)', 'S(8)', 'E(2)%', 'E(4)%', 'E(8)%'])

    for alg in algorithms:
        for lang in languages:
            for wc in word_counts:
                t1 = data[alg][1][lang].get(wc)
                t2 = data[alg][2][lang].get(wc)
                t4 = data[alg][4][lang].get(wc)
                t8 = data[alg][8][lang].get(wc)

                s2 = calc_speedup(t1, t2)
                s4 = calc_speedup(t1, t4)
                s8 = calc_speedup(t1, t8)

                e2 = calc_efficiency(s2, 2) * 100 if s2 else None
                e4 = calc_efficiency(s4, 4) * 100 if s4 else None
                e8 = calc_efficiency(s8, 8) * 100 if s8 else None

                writer.writerow([
                    alg, lang, wc,
                    f"{t1:.2f}" if t1 else "N/A",
                    f"{t2:.2f}" if t2 else "N/A",
                    f"{t4:.2f}" if t4 else "N/A",
                    f"{t8:.2f}" if t8 else "N/A",
                    f"{s2:.2f}" if s2 else "N/A",
                    f"{s4:.2f}" if s4 else "N/A",
                    f"{s8:.2f}" if s8 else "N/A",
                    f"{e2:.1f}" if e2 else "N/A",
                    f"{e4:.1f}" if e4 else "N/A",
                    f"{e8:.1f}" if e8 else "N/A",
                ])

print(f"Saved: {csv_path}")

# Save summary CSV
summary_path = output_dir / 'speedup_efficiency_summary_text_split.csv'
with open(summary_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    writer.writerow(['Algorithm', 'Language', 'Avg_S(2)', 'Avg_S(4)', 'Avg_S(8)',
                     'Avg_E(2)%', 'Avg_E(4)%', 'Avg_E(8)%'])

    for alg in algorithms:
        for lang in languages:
            speedups = {2: [], 4: [], 8: []}

            for wc in word_counts:
                t1 = data[alg][1][lang].get(wc)
                for p in [2, 4, 8]:
                    tp = data[alg][p][lang].get(wc)
                    s = calc_speedup(t1, tp)
                    if s is not None:
                        speedups[p].append(s)

            avg_s2 = sum(speedups[2]) / len(speedups[2]) if speedups[2] else None
            avg_s4 = sum(speedups[4]) / len(speedups[4]) if speedups[4] else None
            avg_s8 = sum(speedups[8]) / len(speedups[8]) if speedups[8] else None

            avg_e2 = (avg_s2 / 2 * 100) if avg_s2 else None
            avg_e4 = (avg_s4 / 4 * 100) if avg_s4 else None
            avg_e8 = (avg_s8 / 8 * 100) if avg_s8 else None

            writer.writerow([
                alg, lang,
                f"{avg_s2:.2f}" if avg_s2 else "N/A",
                f"{avg_s4:.2f}" if avg_s4 else "N/A",
                f"{avg_s8:.2f}" if avg_s8 else "N/A",
                f"{avg_e2:.1f}" if avg_e2 else "N/A",
                f"{avg_e4:.1f}" if avg_e4 else "N/A",
                f"{avg_e8:.1f}" if avg_e8 else "N/A",
            ])

print(f"Saved: {summary_path}")
