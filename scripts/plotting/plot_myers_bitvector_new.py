"""
Plot Myers Bit-Vector Scalability - New Results

Visualizes the text-split benchmark results for Myers Bit-Vector algorithm
across different word counts and process counts.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Research paper style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
})

# Define all word count files in order
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

# Data structure: {num_procs: {language: [times for each word count]}}
data = {1: {'MK': [], 'EN': [], 'TR': []}, 2: {'MK': [], 'EN': [], 'TR': []}, 4: {'MK': [], 'EN': [], 'TR': []}, 8: {'MK': [], 'EN': [], 'TR': []}}
word_counts = []

results_dir = Path(__file__).parent.parent.parent  # Go up to project root

for file_path, num_words in file_info:
    full_path = results_dir / file_path
    if not full_path.exists():
        print(f"Warning: {full_path} not found, skipping...")
        continue

    word_counts.append(num_words)
    with open(full_path, 'r') as f:
        results = json.load(f)

    text_split = results.get('text_split', {})

    for num_procs in ['1', '2', '4', '8']:
        if num_procs in text_split:
            for entry in text_split[num_procs]:
                if entry['algorithm'] == 'Myers Bit-Vector':
                    lang = entry['language']
                    time_s = entry['total_time_s']
                    data[int(num_procs)][lang].append(time_s)

# Load Turkish data from separate results file
tr_results_path = results_dir / 'results' / 'text_split' / 'tr_myers_best_results.json'
if tr_results_path.exists():
    with open(tr_results_path, 'r') as f:
        tr_results = json.load(f)

    # Map word counts to expected values (handle slight variations like 2499 vs 2500, 4998 vs 5000)
    word_count_map = {250: 250, 500: 500, 1000: 1000, 1500: 1500, 2000: 2000, 2499: 2500, 2500: 2500, 3500: 3500, 4998: 5000, 5000: 5000}

    for entry in tr_results.get('results', []):
        if entry['algorithm'] in ['Myers-BitVector', 'Myers Bit-Vector']:
            num_procs = entry['num_procs']
            misspelled = entry['misspelled']
            mapped_words = word_count_map.get(misspelled, misspelled)
            if mapped_words in word_counts and num_procs in data:
                idx = word_counts.index(mapped_words)
                while len(data[num_procs]['TR']) < idx:
                    data[num_procs]['TR'].append(None)
                if len(data[num_procs]['TR']) == idx:
                    data[num_procs]['TR'].append(entry['total_time_s'])
else:
    print(f"Warning: Turkish results file not found: {tr_results_path}")

# Create the plot with 3 subplots (one per language)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Colors and markers for different process counts
colors = {1: '#9467BD', 2: '#D62728', 4: '#2CA02C', 8: '#1F77B4'}  # purple, red, green, blue
markers = {1: 'D', 2: 'o', 4: 's', 8: '^'}  # diamond, circle, square, triangle

languages = ['MK', 'EN', 'TR']
lang_names = {'MK': 'Macedonian', 'EN': 'English', 'TR': 'Turkish'}

for idx, lang in enumerate(languages):
    ax = axes[idx]

    for num_procs in [1, 2, 4, 8]:
        times = data[num_procs][lang]
        if len(times) == len(word_counts) and all(t is not None for t in times):
            ax.plot(word_counts, times,
                   color=colors[num_procs],
                   linestyle='-',
                   marker=markers[num_procs],
                   markerfacecolor=colors[num_procs],
                   markeredgecolor='black',
                   markeredgewidth=0.5,
                   label=f'{num_procs}P')

    ax.set_xlabel('Number of Words')
    if idx == 0:
        ax.set_ylabel('Time (seconds)')
    ax.set_title(f'{lang_names[lang]} ({lang})')
    ax.set_xticks(word_counts)
    ax.set_xticklabels([str(w) for w in word_counts], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, max(word_counts) + 250)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='black', fancybox=False)

fig.suptitle('MPI Text-Split Scalability: Myers Bit-Vector Algorithm', fontsize=14, fontweight='bold')

# Tight layout
plt.tight_layout()

# Save the figures
output_dir = results_dir / 'visualizations' / 'new_res'
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / 'myers_bitvector_scalability_new.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(output_dir / 'myers_bitvector_scalability_new.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Also save in main directory
plt.savefig(results_dir / 'myers_bitvector_scalability_new.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"Saved visualizations:")
print(f"  - {output_dir / 'myers_bitvector_scalability_new.png'}")
print(f"  - {output_dir / 'myers_bitvector_scalability_new.pdf'}")
print(f"  - {results_dir / 'myers_bitvector_scalability_new.png'}")

# Print summary statistics
print("\n--- Myers Bit-Vector Performance Summary ---")
print(f"{'Words':<8} {'1P MK':>10} {'1P EN':>10} {'1P TR':>10} {'2P MK':>10} {'2P EN':>10} {'2P TR':>10} {'4P MK':>10} {'4P EN':>10} {'4P TR':>10} {'8P MK':>10} {'8P EN':>10} {'8P TR':>10}")
print("-" * 128)
for i, wc in enumerate(word_counts):
    row = f"{wc:<8}"
    for np in [1, 2, 4, 8]:
        for lang in ['MK', 'EN', 'TR']:
            if i < len(data[np][lang]) and data[np][lang][i] is not None:
                row += f"{data[np][lang][i]:>10.2f}"
            else:
                row += f"{'N/A':>10}"
    print(row)
