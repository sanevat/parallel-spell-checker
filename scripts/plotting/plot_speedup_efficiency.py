"""
Plot Speedup and Efficiency for all algorithms, word counts, and process counts.
Creates comprehensive comparison charts for Text-Split and Dict-Split strategies.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Research paper style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

BASE = Path(__file__).parent.parent.parent

def load_results(strategy='text_split'):
    """Load all results for a strategy."""
    if strategy == 'text_split':
        results_dir = BASE / 'results' / 'text_split'
        key = 'text_split'
    else:
        results_dir = BASE / 'results' / 'dict_split'
        key = 'dict_split'

    data = {}

    for json_file in sorted(results_dir.glob('*.json')):
        # Extract word count from filename
        name = json_file.stem
        if '250' in name:
            words = 250
        elif '500' in name and '1500' not in name and '2500' not in name and '3500' not in name and '5000' not in name:
            words = 500
        elif '1K' in name:
            words = 1000
        elif '1500' in name:
            words = 1500
        elif '2K' in name:
            words = 2000
        elif '2500' in name:
            words = 2500
        elif '3500' in name:
            words = 3500
        elif '5000' in name:
            words = 5000
        else:
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            content = json.load(f)

        ts = content.get(key, {})

        # Collect times by algorithm, language, and procs
        for procs_str, entries in ts.items():
            procs = int(procs_str)
            for entry in entries:
                algo = entry['algorithm']
                lang = entry['language']
                time_s = entry['total_time_s']

                k = (algo, lang)
                if k not in data:
                    data[k] = {}
                if words not in data[k]:
                    data[k][words] = {}
                data[k][words][procs] = time_s

    return data


def calculate_speedup_efficiency(data):
    """Calculate speedup and efficiency from time data."""
    results = {}

    for (algo, lang), word_data in data.items():
        results[(algo, lang)] = {}
        for words, proc_times in word_data.items():
            t1 = proc_times.get(1, None)
            if t1 is None:
                continue

            results[(algo, lang)][words] = {}
            for procs, time in proc_times.items():
                speedup = t1 / time if time > 0 else 0
                efficiency = (speedup / procs) * 100
                results[(algo, lang)][words][procs] = {
                    'time': time,
                    'speedup': speedup,
                    'efficiency': efficiency
                }

    return results


def plot_speedup_comparison(text_data, dict_data, output_dir):
    """Plot speedup comparison for all algorithms."""

    word_counts = [250, 500, 1000, 1500, 2000, 2500, 3500, 5000]
    procs_list = [2, 4, 8]

    # Get unique algorithms
    text_algos = set(algo for algo, lang in text_data.keys())
    dict_algos = set(algo for algo, lang in dict_data.keys())

    colors = {'Levenshtein': '#1f77b4', 'Damerau-Levenshtein': '#d62728', 'Myers Bit-Vector': '#2ca02c'}
    markers = {2: 'o', 4: 's', 8: '^'}

    # Create figure with 2 rows (Text-Split, Dict-Split) x 3 cols (2P, 4P, 8P)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    strategies = [('Text-Split', text_data), ('Dict-Split', dict_data)]

    for row, (strategy_name, data) in enumerate(strategies):
        for col, procs in enumerate(procs_list):
            ax = axes[row, col]

            for (algo, lang), word_results in sorted(data.items()):
                if lang != 'MK':  # Only plot MK for clarity
                    continue

                x = []
                y = []
                for words in word_counts:
                    if words in word_results and procs in word_results[words]:
                        x.append(words)
                        y.append(word_results[words][procs]['speedup'])

                if x:
                    color = colors.get(algo, 'gray')
                    ax.plot(x, y, marker=markers[procs], color=color,
                           label=algo, linewidth=1.5, markersize=6)

            # Ideal speedup line
            ax.axhline(y=procs, color='gray', linestyle='--', alpha=0.5, label=f'Ideal ({procs})')

            ax.set_xlabel('Number of Words')
            ax.set_ylabel('Speedup')
            ax.set_title(f'{strategy_name} - {procs} Processes')
            ax.set_xticks(word_counts)
            ax.set_xticklabels([str(w) for w in word_counts], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, procs + 1)

            if row == 0 and col == 2:
                ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison_all.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'speedup_comparison_all.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_dir / 'speedup_comparison_all.png'}")


def plot_efficiency_comparison(text_data, dict_data, output_dir):
    """Plot efficiency comparison for all algorithms."""

    word_counts = [250, 500, 1000, 1500, 2000, 2500, 3500, 5000]
    procs_list = [2, 4, 8]

    colors = {'Levenshtein': '#1f77b4', 'Damerau-Levenshtein': '#d62728', 'Myers Bit-Vector': '#2ca02c'}
    markers = {2: 'o', 4: 's', 8: '^'}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    strategies = [('Text-Split', text_data), ('Dict-Split', dict_data)]

    for row, (strategy_name, data) in enumerate(strategies):
        for col, procs in enumerate(procs_list):
            ax = axes[row, col]

            for (algo, lang), word_results in sorted(data.items()):
                if lang != 'MK':
                    continue

                x = []
                y = []
                for words in word_counts:
                    if words in word_results and procs in word_results[words]:
                        x.append(words)
                        y.append(word_results[words][procs]['efficiency'])

                if x:
                    color = colors.get(algo, 'gray')
                    ax.plot(x, y, marker=markers[procs], color=color,
                           label=algo, linewidth=1.5, markersize=6)

            # 100% efficiency line
            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal (100%)')

            ax.set_xlabel('Number of Words')
            ax.set_ylabel('Efficiency (%)')
            ax.set_title(f'{strategy_name} - {procs} Processes')
            ax.set_xticks(word_counts)
            ax.set_xticklabels([str(w) for w in word_counts], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 120)

            if row == 0 and col == 2:
                ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_comparison_all.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'efficiency_comparison_all.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_dir / 'efficiency_comparison_all.png'}")


def plot_combined_heatmap(text_data, dict_data, output_dir):
    """Plot heatmaps showing speedup for all combinations."""

    word_counts = [250, 500, 1000, 1500, 2000, 2500, 3500, 5000]
    procs_list = [2, 4, 8]

    # Collect all algorithms
    all_algos = sorted(set(algo for algo, lang in text_data.keys()))

    fig, axes = plt.subplots(2, len(all_algos), figsize=(5*len(all_algos), 10))

    strategies = [('Text-Split', text_data), ('Dict-Split', dict_data)]

    for row, (strategy_name, data) in enumerate(strategies):
        for col, algo in enumerate(all_algos):
            ax = axes[row, col] if len(all_algos) > 1 else axes[row]

            # Create matrix for heatmap (words x procs)
            matrix = np.zeros((len(word_counts), len(procs_list)))

            key = (algo, 'MK')
            if key in data:
                for i, words in enumerate(word_counts):
                    if words in data[key]:
                        for j, procs in enumerate(procs_list):
                            if procs in data[key][words]:
                                matrix[i, j] = data[key][words][procs]['speedup']

            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=8)

            # Add text annotations
            for i in range(len(word_counts)):
                for j in range(len(procs_list)):
                    val = matrix[i, j]
                    color = 'white' if val > 4 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

            ax.set_xticks(range(len(procs_list)))
            ax.set_xticklabels([f'{p}P' for p in procs_list])
            ax.set_yticks(range(len(word_counts)))
            ax.set_yticklabels(word_counts)
            ax.set_xlabel('Processes')
            ax.set_ylabel('Words')
            ax.set_title(f'{strategy_name}\n{algo}')

    plt.colorbar(im, ax=axes.ravel().tolist(), label='Speedup', shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_heatmap_all.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'speedup_heatmap_all.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_dir / 'speedup_heatmap_all.png'}")


def plot_strategy_comparison(text_data, dict_data, output_dir):
    """Plot direct comparison between Text-Split and Dict-Split for each algorithm."""

    word_counts = [250, 500, 1000, 1500, 2000, 2500, 3500, 5000]

    # Common algorithms in both
    text_algos = set(algo for algo, lang in text_data.keys())
    dict_algos = set(algo for algo, lang in dict_data.keys())
    common_algos = sorted(text_algos & dict_algos)

    fig, axes = plt.subplots(len(common_algos), 2, figsize=(12, 4*len(common_algos)))

    for row, algo in enumerate(common_algos):
        # Speedup plot
        ax1 = axes[row, 0] if len(common_algos) > 1 else axes[0]
        # Efficiency plot
        ax2 = axes[row, 1] if len(common_algos) > 1 else axes[1]

        for procs in [2, 4, 8]:
            # Text-Split
            x_ts, y_ts_s, y_ts_e = [], [], []
            key = (algo, 'MK')
            if key in text_data:
                for words in word_counts:
                    if words in text_data[key] and procs in text_data[key][words]:
                        x_ts.append(words)
                        y_ts_s.append(text_data[key][words][procs]['speedup'])
                        y_ts_e.append(text_data[key][words][procs]['efficiency'])

            # Dict-Split
            x_ds, y_ds_s, y_ds_e = [], [], []
            if key in dict_data:
                for words in word_counts:
                    if words in dict_data[key] and procs in dict_data[key][words]:
                        x_ds.append(words)
                        y_ds_s.append(dict_data[key][words][procs]['speedup'])
                        y_ds_e.append(dict_data[key][words][procs]['efficiency'])

            color = {2: '#1f77b4', 4: '#2ca02c', 8: '#d62728'}[procs]

            if x_ts:
                ax1.plot(x_ts, y_ts_s, '-', marker='o', color=color, label=f'Text-Split {procs}P')
                ax2.plot(x_ts, y_ts_e, '-', marker='o', color=color, label=f'Text-Split {procs}P')
            if x_ds:
                ax1.plot(x_ds, y_ds_s, '--', marker='s', color=color, label=f'Dict-Split {procs}P')
                ax2.plot(x_ds, y_ds_e, '--', marker='s', color=color, label=f'Dict-Split {procs}P')

        ax1.set_xlabel('Number of Words')
        ax1.set_ylabel('Speedup')
        ax1.set_title(f'{algo} - Speedup')
        ax1.set_xticks(word_counts)
        ax1.set_xticklabels([str(w) for w in word_counts], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=7, ncol=2)

        ax2.set_xlabel('Number of Words')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title(f'{algo} - Efficiency')
        ax2.set_xticks(word_counts)
        ax2.set_xticklabels([str(w) for w in word_counts], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
        ax2.legend(loc='best', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_comparison_all.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'strategy_comparison_all.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_dir / 'strategy_comparison_all.png'}")


def main():
    print("=" * 60)
    print("SPEEDUP & EFFICIENCY VISUALIZATION")
    print("=" * 60)

    # Load data
    print("\nLoading Text-Split results...")
    text_raw = load_results('text_split')
    text_data = calculate_speedup_efficiency(text_raw)
    print(f"  Loaded {len(text_data)} algorithm/language combinations")

    print("\nLoading Dict-Split results...")
    dict_raw = load_results('dict_split')
    dict_data = calculate_speedup_efficiency(dict_raw)
    print(f"  Loaded {len(dict_data)} algorithm/language combinations")

    # Output directory
    output_dir = BASE / 'visualizations' / 'speedup_efficiency'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    plot_speedup_comparison(text_data, dict_data, output_dir)
    plot_efficiency_comparison(text_data, dict_data, output_dir)
    plot_combined_heatmap(text_data, dict_data, output_dir)
    plot_strategy_comparison(text_data, dict_data, output_dir)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
