"""
Plot Levenshtein Scalability - Dict-Split Results

Visualizes the dict-split benchmark results for Levenshtein algorithm
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
    ('results/dict_split/dict_split_250.json', 250),
    ('results/dict_split/dict_split_500.json', 500),
    ('results/dict_split/dict_split_1K.json', 1000),
    ('results/dict_split/dict_split_1500.json', 1500),
    ('results/dict_split/dict_split_2K.json', 2000),
    ('results/dict_split/dict_split_2500.json', 2500),
    ('results/dict_split/dict_split_3500.json', 3500),
    ('results/dict_split/dict_split_5000.json', 5000),
]

# Data structure: {num_procs: {language: [times for each word count]}}
data = {1: {'MK': [], 'EN': []}, 2: {'MK': [], 'EN': []}, 4: {'MK': [], 'EN': []}, 8: {'MK': [], 'EN': []}}
word_counts = []

results_dir = Path(__file__).parent

for file_path, num_words in file_info:
    full_path = results_dir / file_path
    if not full_path.exists():
        print(f"Warning: {full_path} not found, skipping...")
        continue

    word_counts.append(num_words)
    with open(full_path, 'r') as f:
        results = json.load(f)

    dict_split = results.get('dict_split', {})

    for num_procs in ['1', '2', '4', '8']:
        if num_procs in dict_split:
            for entry in dict_split[num_procs]:
                if entry['algorithm'] == 'Levenshtein':
                    lang = entry['language']
                    time_s = entry['total_time_s']
                    data[int(num_procs)][lang].append(time_s)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Colors and markers for different process counts
colors = {1: '#9467BD', 2: '#D62728', 4: '#2CA02C', 8: '#1F77B4'}  # purple, red, green, blue
markers = {1: 'D', 2: 'o', 4: 's', 8: '^'}  # diamond, circle, square, triangle

# Plot MK (solid lines)
for num_procs in [1, 2, 4, 8]:
    times = data[num_procs]['MK']
    if len(times) == len(word_counts):
        ax.plot(word_counts, times,
               color=colors[num_procs],
               linestyle='-',
               marker=markers[num_procs],
               markerfacecolor=colors[num_procs],
               markeredgecolor='black',
               markeredgewidth=0.5,
               label=f'{num_procs} Process{"es" if num_procs > 1 else ""} (MK)')

# Plot EN (dashed lines)
for num_procs in [1, 2, 4, 8]:
    times = data[num_procs]['EN']
    if len(times) == len(word_counts):
        ax.plot(word_counts, times,
               color=colors[num_procs],
               linestyle='--',
               marker=markers[num_procs],
               markerfacecolor='white',
               markeredgecolor=colors[num_procs],
               markeredgewidth=1.5,
               label=f'{num_procs} Process{"es" if num_procs > 1 else ""} (EN)')

# Configure the plot
ax.set_xlabel('Number of Words')
ax.set_ylabel('Time (seconds)')
ax.set_title('MPI Dict-Split Scalability: Levenshtein Algorithm')

# Set x-axis ticks
ax.set_xticks(word_counts)
ax.set_xticklabels([str(w) for w in word_counts], rotation=45, ha='right')

# Create custom legend with clear distinction
from matplotlib.lines import Line2D

legend_elements = [
    # Language style entries
    Line2D([0], [0], color='gray', linestyle='-', marker='o',
           markerfacecolor='gray', markeredgecolor='black', markeredgewidth=0.5,
           label='MK (solid, filled)'),
    Line2D([0], [0], color='gray', linestyle='--', marker='o',
           markerfacecolor='white', markeredgecolor='gray', markeredgewidth=1.5,
           label='EN (dashed, hollow)'),
    Line2D([0], [0], color='none', label=''),  # spacer
    # Process count entries
    Line2D([0], [0], color=colors[1], linestyle='-', marker='D',
           markerfacecolor=colors[1], markeredgecolor='black', markeredgewidth=0.5,
           label='1 Process'),
    Line2D([0], [0], color=colors[2], linestyle='-', marker='o',
           markerfacecolor=colors[2], markeredgecolor='black', markeredgewidth=0.5,
           label='2 Processes'),
    Line2D([0], [0], color=colors[4], linestyle='-', marker='s',
           markerfacecolor=colors[4], markeredgecolor='black', markeredgewidth=0.5,
           label='4 Processes'),
    Line2D([0], [0], color=colors[8], linestyle='-', marker='^',
           markerfacecolor=colors[8], markeredgecolor='black', markeredgewidth=0.5,
           label='8 Processes'),
]

ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95,
          edgecolor='black', fancybox=False)

# Add grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Set axis limits with some padding
ax.set_xlim(0, max(word_counts) + 250)
ax.set_ylim(0, None)

# Tight layout
plt.tight_layout()

# Save the figures
output_dir = results_dir / 'visualizations' / 'dict_split'
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / 'levenshtein_dictsplit_scalability.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(output_dir / 'levenshtein_dictsplit_scalability.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Also save in main directory
plt.savefig(results_dir / 'levenshtein_dictsplit_scalability.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"Saved visualizations:")
print(f"  - {output_dir / 'levenshtein_dictsplit_scalability.png'}")
print(f"  - {output_dir / 'levenshtein_dictsplit_scalability.pdf'}")
print(f"  - {results_dir / 'levenshtein_dictsplit_scalability.png'}")

# Print summary statistics
print("\n--- Levenshtein Dict-Split Performance Summary ---")
print(f"{'Words':<8} {'1P MK':>12} {'1P EN':>12} {'2P MK':>12} {'2P EN':>12} {'4P MK':>12} {'4P EN':>12} {'8P MK':>12} {'8P EN':>12}")
print("-" * 104)
for i, wc in enumerate(word_counts):
    row = f"{wc:<8}"
    for np in [1, 2, 4, 8]:
        for lang in ['MK', 'EN']:
            if i < len(data[np][lang]):
                row += f"{data[np][lang][i]:>12.2f}"
            else:
                row += f"{'N/A':>12}"
    print(row)
