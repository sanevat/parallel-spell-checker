"""
Plot All Languages Comparison - Text-Split and Dict-Split

Generates visualizations with 3 subplots side by side (MK, EN, TR)
for each algorithm and parallelization technique.
"""

import pandas as pd
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
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 8,
})

# Paths
script_dir = Path(__file__).parent
project_dir = script_dir.parent.parent
output_dir = project_dir / 'visualizations' / 'added_tr'
output_dir.mkdir(parents=True, exist_ok=True)

# Read CSV files
text_split_csv = project_dir / 'results' / 'text_split' / 'speedup_efficiency_text_split.csv'
dict_split_csv = project_dir / 'results' / 'dict_split' / 'speedup_efficiency_dict_split.csv'

df_text = pd.read_csv(text_split_csv)
df_dict = pd.read_csv(dict_split_csv)

# Colors and markers for different process counts (matching the example image)
colors = {1: '#9467BD', 2: '#D62728', 4: '#2CA02C', 8: '#1F77B4'}  # purple, red, green, blue
markers = {1: 'D', 2: 'o', 4: 's', 8: '^'}  # diamond, circle, square, triangle
labels = {1: '1P', 2: '2P', 4: '4P', 8: '8P'}

# Language full names
lang_names = {'MK': 'Macedonian (MK)', 'EN': 'English (EN)', 'TR': 'Turkish (TR)'}

algorithms = ['Levenshtein', 'Damerau-Levenshtein', 'Myers Bit-Vector']
techniques = [('text_split', df_text, 'Text-Split'), ('dict_split', df_dict, 'Dict-Split')]

def create_plot(df, algorithm, technique_name, technique_label):
    """Create a plot with 3 subplots side by side for MK, EN, TR."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Main title
    fig.suptitle(f'MPI {technique_label} Scalability: {algorithm} Algorithm', fontsize=14, fontweight='bold')

    # Filter data for this algorithm
    df_alg = df[df['Algorithm'] == algorithm]
    word_counts = sorted(df_alg['Words'].unique())

    # Find max time for consistent y-axis
    max_time = 0
    for lang in ['MK', 'EN', 'TR']:
        df_lang = df_alg[df_alg['Language'] == lang]
        if not df_lang.empty and 'T(1)' in df_lang.columns:
            max_time = max(max_time, df_lang['T(1)'].max())

    # Plot each language in its own subplot
    for idx, lang in enumerate(['MK', 'EN', 'TR']):
        ax = axes[idx]
        df_lang = df_alg[df_alg['Language'] == lang]

        if df_lang.empty:
            ax.set_title(lang_names[lang])
            continue

        for num_procs in [1, 2, 4, 8]:
            times = []
            valid_word_counts = []

            for wc in word_counts:
                row = df_lang[df_lang['Words'] == wc]
                if not row.empty:
                    time_col = f'T({num_procs})'
                    if time_col in row.columns:
                        times.append(row[time_col].values[0])
                        valid_word_counts.append(wc)

            if times:
                ax.plot(valid_word_counts, times,
                       color=colors[num_procs],
                       linestyle='-',
                       marker=markers[num_procs],
                       markerfacecolor=colors[num_procs],
                       markeredgecolor='black',
                       markeredgewidth=0.5,
                       label=labels[num_procs])

        # Configure subplot
        ax.set_title(lang_names[lang])
        ax.set_xlabel('Number of Words')
        if idx == 0:
            ax.set_ylabel('Time (seconds)')

        # Set x-axis ticks
        ax.set_xticks(word_counts)
        ax.set_xticklabels([str(w) for w in word_counts], rotation=45, ha='right')

        # Add legend
        ax.legend(loc='upper left', framealpha=0.95, edgecolor='black', fancybox=False)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Set axis limits
        ax.set_xlim(min(word_counts) - 100, max(word_counts) + 100)

    # Set consistent y-axis
    axes[0].set_ylim(0, max_time * 1.1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig

# Generate all plots
print("Generating visualizations...")
print(f"Output directory: {output_dir}")
print()

for algorithm in algorithms:
    alg_short = algorithm.lower().replace(' ', '_').replace('-', '_')

    for technique_name, df, technique_label in techniques:
        fig = create_plot(df, algorithm, technique_name, technique_label)
        filename = f'{alg_short}_{technique_name}.png'
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"Saved: {filename}")

print()
print(f"All visualizations saved to: {output_dir}")
print(f"Total files generated: {len(algorithms) * len(techniques)}")
