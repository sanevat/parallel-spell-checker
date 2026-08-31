import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

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

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(script_dir))

# Define word count files
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

# Algorithm name mapping
algorithm_map = {
    'Levenshtein': ['Levenshtein'],
    'Damerau-Levenshtein': ['Damerau-Levenshtein'],
    'Myers-BitVector': ['Myers Bit-Vector', 'Myers-BitVector'],
}

# Word count mapping for TR data
word_count_map = {249: 250, 498: 500, 999: 1000, 1498: 1500, 1997: 2000,
                  2498: 2500, 2499: 2500, 3497: 3500, 4998: 5000, 5000: 5000,
                  250: 250, 500: 500, 1000: 1000, 1500: 1500, 2000: 2000,
                  2500: 2500, 3500: 3500}

def load_mk_en_data(algorithm_names):
    """Load MK and EN data from text_split files."""
    data = {1: {'MK': [], 'EN': []}, 2: {'MK': [], 'EN': []},
            4: {'MK': [], 'EN': []}, 8: {'MK': [], 'EN': []}}
    word_counts = []

    for file_path, num_words in file_info:
        full_path = os.path.join(base_dir, file_path)
        word_counts.append(num_words)

        with open(full_path, 'r') as f:
            results = json.load(f)

        text_split = results.get('text_split', {})

        for num_procs_str in ['1', '2', '4', '8']:
            num_procs = int(num_procs_str)
            if num_procs_str in text_split:
                for entry in text_split[num_procs_str]:
                    if entry['algorithm'] in algorithm_names:
                        lang = entry['language']
                        if lang in ['MK', 'EN']:
                            time_s = entry.get('total_time_s', entry['ms_per_word'] * num_words / 1000)
                            # Only add if we haven't added this word count yet
                            if len(data[num_procs][lang]) < len(word_counts):
                                data[num_procs][lang].append(time_s)

    return data, word_counts

def load_tr_data(tr_file, algorithm_names):
    """Load TR data from separate results file."""
    tr_path = os.path.join(base_dir, tr_file)
    data = {1: [], 2: [], 4: [], 8: []}

    if not os.path.exists(tr_path):
        return data

    with open(tr_path, 'r') as f:
        tr_results = json.load(f)

    # Organize by num_procs and word count
    tr_by_procs = {1: {}, 2: {}, 4: {}, 8: {}}

    for entry in tr_results.get('results', []):
        if entry['algorithm'] in algorithm_names:
            num_procs = entry['num_procs']
            if num_procs in [1, 2, 4, 8]:
                misspelled = entry['misspelled']
                mapped_words = word_count_map.get(misspelled, misspelled)
                tr_by_procs[num_procs][mapped_words] = entry['total_time_s']

    # Convert to ordered list
    word_counts = [250, 500, 1000, 1500, 2000, 2500, 3500, 5000]
    for num_procs in [1, 2, 4, 8]:
        for wc in word_counts:
            if wc in tr_by_procs[num_procs]:
                data[num_procs].append(tr_by_procs[num_procs][wc])
            else:
                data[num_procs].append(None)

    return data

def create_algorithm_plot(algorithm_key, algorithm_names, tr_file, output_name, title):
    """Create a 3-subplot figure for one algorithm."""
    # Load data
    mk_en_data, word_counts = load_mk_en_data(algorithm_names)
    tr_data = load_tr_data(tr_file, algorithm_names)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    # Colors and markers for different process counts
    colors = {1: '#9467BD', 2: '#D62728', 4: '#2CA02C', 8: '#1F77B4'}  # purple, red, green, blue
    markers = {1: 'D', 2: 'o', 4: 's', 8: '^'}  # diamond, circle, square, triangle
    labels = {1: '1P', 2: '2P', 4: '4P', 8: '8P'}

    languages = [('MK', 'Macedonian (MK)', mk_en_data),
                 ('EN', 'English (EN)', mk_en_data),
                 ('TR', 'Turkish (TR)', None)]

    for ax_idx, (lang_code, lang_name, data_source) in enumerate(languages):
        ax = axes[ax_idx]

        for num_procs in [1, 2, 4, 8]:
            if lang_code == 'TR':
                times = tr_data[num_procs]
            else:
                times = data_source[num_procs][lang_code]

            # Filter out None values
            valid_times = []
            valid_wc = []
            for i, t in enumerate(times):
                if t is not None and i < len(word_counts):
                    valid_times.append(t)
                    valid_wc.append(word_counts[i])

            if len(valid_times) > 0:
                ax.plot(valid_wc, valid_times,
                       color=colors[num_procs],
                       linestyle='-',
                       marker=markers[num_procs],
                       markerfacecolor=colors[num_procs],
                       markeredgecolor='black',
                       markeredgewidth=0.5,
                       label=labels[num_procs])

        ax.set_xlabel('Number of Words')
        ax.set_title(lang_name)
        ax.set_xticks(word_counts)
        ax.set_xticklabels(word_counts, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(0, 5250)
        ax.set_ylim(0, None)

        if ax_idx == 0:
            ax.set_ylabel('Time (seconds)')
            ax.legend(loc='upper left', framealpha=0.95, edgecolor='black', fancybox=False)

    # Main title
    fig.suptitle(f'MPI Text-Split Scalability: {title}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save the figure
    output_dir = os.path.join(base_dir, 'visualizations', 'added_tr')
    os.makedirs(output_dir, exist_ok=True)

    png_path = os.path.join(output_dir, f'{output_name}.png')
    pdf_path = os.path.join(output_dir, f'{output_name}.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

# Generate plots for all three algorithms
algorithms = [
    ('Levenshtein', ['Levenshtein'],
     'results/text_split/tr_levenshtein_best_results.json',
     'text_split_levenshtein', 'Levenshtein Algorithm'),

    ('Damerau-Levenshtein', ['Damerau-Levenshtein'],
     'results/text_split/tr_damerau_best_results.json',
     'text_split_damerau_levenshtein', 'Damerau-Levenshtein Algorithm'),

    ('Myers-BitVector', ['Myers Bit-Vector', 'Myers-BitVector'],
     'results/text_split/tr_myers_best_results.json',
     'text_split_myers_bitvector', 'Myers Bit-Vector Algorithm'),
]

for algo_key, algo_names, tr_file, output_name, title in algorithms:
    print(f"\nGenerating plot for {algo_key}...")
    create_algorithm_plot(algo_key, algo_names, tr_file, output_name, title)

print("\nAll plots generated successfully!")
