"""
Generate MPI scalability plots (MK + EN + TR) for all algorithms and both
split strategies (dict-split, text-split). Output to visualizations/added_tr/.

Style matches the Damerau-Levenshtein Dict-Split reference plot:
    - 1/2/4/8 processes (purple/red/green/blue with diamond/circle/square/triangle)
    - MK solid + filled markers
    - EN dashed + hollow markers
    - TR dotted  + filled markers (smaller)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / 'visualizations' / 'added_tr'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


WORD_COUNTS = [250, 500, 1000, 1500, 2000, 2500, 3500, 5000]

DICT_SPLIT_FILES = {
    250:  'dict_split_250.json',
    500:  'dict_split_500.json',
    1000: 'dict_split_1K.json',
    1500: 'dict_split_1500.json',
    2000: 'dict_split_2K.json',
    2500: 'dict_split_2500.json',
    3500: 'dict_split_3500.json',
    5000: 'dict_split_5000.json',
}

TEXT_SPLIT_FILES = {
    250:  'text_split_250.json',
    500:  'text_split_500.json',
    1000: 'text_split_1K.json',
    1500: 'text_split_1500.json',
    2000: 'text_split_2K.json',
    2500: 'text_split_2500.json',
    3500: 'text_split_3500.json',
    5000: 'text_split_5000.json',
}

ALGORITHMS = [
    ('Damerau-Levenshtein', {'Damerau-Levenshtein'}),
    ('Levenshtein',         {'Levenshtein'}),
    ('Myers Bit-Vector',    {'Myers Bit-Vector', 'Myers-BitVector'}),
]

TR_DICT_SPLIT_FILES = {
    'Damerau-Levenshtein': 'tr_damerau_dict_split_best_results.json',
    'Levenshtein':         'tr_levenshtein_dict_split_best_results.json',
    'Myers Bit-Vector':    'tr_myers_dict_split_best_results.json',
}

TR_TEXT_SPLIT_FILES = {
    'Damerau-Levenshtein': 'tr_damerau_best_results.json',
    'Levenshtein':         'tr_levenshtein_best_results.json',
    'Myers Bit-Vector':    'tr_myers_best_results.json',
}

# JSON "misspelled" counts -> canonical word-count bucket
MISSPELLED_MAP = {
    249: 250, 250: 250,
    498: 500, 500: 500,
    999: 1000, 1000: 1000,
    1498: 1500, 1500: 1500,
    1997: 2000, 2000: 2000,
    2498: 2500, 2499: 2500, 2500: 2500,
    3497: 3500, 3500: 3500,
    4998: 5000, 5000: 5000,
}


def load_mk_en(split_key, files, results_subdir, algo_names):
    """Return {num_procs: {'MK': [...], 'EN': [...]}} parallel to WORD_COUNTS."""
    data = {p: {'MK': [None] * len(WORD_COUNTS), 'EN': [None] * len(WORD_COUNTS)}
            for p in (1, 2, 4, 8)}

    for idx, wc in enumerate(WORD_COUNTS):
        fname = files[wc]
        fpath = REPO_ROOT / 'results' / results_subdir / fname
        if not fpath.exists():
            # fall back to legacy layout (no subdir)
            fpath = REPO_ROOT / 'results' / fname
        if not fpath.exists():
            print(f"  [warn] missing {fpath}")
            continue

        with open(fpath, 'r') as f:
            blob = json.load(f)
        section = blob.get(split_key, {})
        for proc_str in ('1', '2', '4', '8'):
            for entry in section.get(proc_str, []):
                if entry.get('algorithm') not in algo_names:
                    continue
                lang = entry.get('language')
                if lang not in ('MK', 'EN'):
                    continue
                t = entry.get('total_time_s')
                if t is None and 'ms_per_word' in entry:
                    t = entry['ms_per_word'] * wc / 1000.0
                data[int(proc_str)][lang][idx] = t
    return data


def load_tr(tr_fname, results_subdir, algo_names):
    """Return {num_procs: [times parallel to WORD_COUNTS]}"""
    data = {p: [None] * len(WORD_COUNTS) for p in (1, 2, 4, 8)}
    fpath = REPO_ROOT / 'results' / results_subdir / tr_fname
    if not fpath.exists():
        print(f"  [warn] missing TR file {fpath}")
        return data

    with open(fpath, 'r') as f:
        blob = json.load(f)
    for entry in blob.get('results', []):
        if entry.get('algorithm') not in algo_names:
            continue
        p = entry.get('num_procs')
        if p not in (1, 2, 4, 8):
            continue
        bucket = MISSPELLED_MAP.get(entry.get('misspelled'))
        if bucket not in WORD_COUNTS:
            continue
        idx = WORD_COUNTS.index(bucket)
        data[p][idx] = entry.get('total_time_s')
    return data


COLORS = {1: '#9467BD', 2: '#D62728', 4: '#2CA02C', 8: '#1F77B4'}
MARKERS = {1: 'D', 2: 'o', 4: 's', 8: '^'}


def plot_series(ax, xs, ys, color, marker, linestyle, filled, smaller=False):
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        return
    xv, yv = zip(*pairs)
    ax.plot(xv, yv,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor=color if filled else 'white',
            markeredgecolor='black' if filled else color,
            markeredgewidth=0.5 if filled else 1.5,
            markersize=5 if smaller else 7)


def build_plot(title, outfile_stem, mken_data, tr_data):
    fig, ax = plt.subplots(figsize=(10, 6))

    for p in (1, 2, 4, 8):
        plot_series(ax, WORD_COUNTS, mken_data[p]['MK'],
                    COLORS[p], MARKERS[p], '-', filled=True)
    for p in (1, 2, 4, 8):
        plot_series(ax, WORD_COUNTS, mken_data[p]['EN'],
                    COLORS[p], MARKERS[p], '--', filled=False)
    for p in (1, 2, 4, 8):
        plot_series(ax, WORD_COUNTS, tr_data[p],
                    COLORS[p], MARKERS[p], ':', filled=True, smaller=True)

    ax.set_xlabel('Number of Words')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.set_xticks(WORD_COUNTS)
    ax.set_xticklabels([str(w) for w in WORD_COUNTS], rotation=45, ha='right')

    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='-', marker='o',
               markerfacecolor='gray', markeredgecolor='black',
               markeredgewidth=0.5, label='MK (solid, filled)'),
        Line2D([0], [0], color='gray', linestyle='--', marker='o',
               markerfacecolor='white', markeredgecolor='gray',
               markeredgewidth=1.5, label='EN (dashed, hollow)'),
        Line2D([0], [0], color='gray', linestyle=':', marker='o',
               markerfacecolor='gray', markeredgecolor='black',
               markeredgewidth=0.5, markersize=5,
               label='TR (dotted, filled)'),
        Line2D([0], [0], color='none', label=''),
        Line2D([0], [0], color=COLORS[1], linestyle='-', marker='D',
               markerfacecolor=COLORS[1], markeredgecolor='black',
               markeredgewidth=0.5, label='1 Process'),
        Line2D([0], [0], color=COLORS[2], linestyle='-', marker='o',
               markerfacecolor=COLORS[2], markeredgecolor='black',
               markeredgewidth=0.5, label='2 Processes'),
        Line2D([0], [0], color=COLORS[4], linestyle='-', marker='s',
               markerfacecolor=COLORS[4], markeredgecolor='black',
               markeredgewidth=0.5, label='4 Processes'),
        Line2D([0], [0], color=COLORS[8], linestyle='-', marker='^',
               markerfacecolor=COLORS[8], markeredgecolor='black',
               markeredgewidth=0.5, label='8 Processes'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              framealpha=0.95, edgecolor='black', fancybox=False)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, max(WORD_COUNTS) + 250)
    ax.set_ylim(0, None)

    plt.tight_layout()
    png = OUTPUT_DIR / f'{outfile_stem}.png'
    pdf = OUTPUT_DIR / f'{outfile_stem}.pdf'
    plt.savefig(png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  wrote {png.name} / {pdf.name}")


def main():
    stem_overrides = {'Myers Bit-Vector': 'myers_bitvector'}
    for algo_title, algo_names in ALGORITHMS:
        stem_algo = stem_overrides.get(
            algo_title,
            algo_title.lower().replace(' ', '_').replace('-', '_'),
        )

        print(f"[dict-split] {algo_title}")
        mken = load_mk_en('dict_split', DICT_SPLIT_FILES, 'dict_split', algo_names)
        tr   = load_tr(TR_DICT_SPLIT_FILES[algo_title], 'dict_split', algo_names)
        build_plot(
            f'MPI Dict-Split Scalability: {algo_title} Algorithm',
            f'dict_split_{stem_algo}',
            mken, tr,
        )

        print(f"[text-split] {algo_title}")
        mken = load_mk_en('text_split', TEXT_SPLIT_FILES, 'text_split', algo_names)
        tr   = load_tr(TR_TEXT_SPLIT_FILES[algo_title], 'text_split', algo_names)
        build_plot(
            f'MPI Text-Split Scalability: {algo_title} Algorithm',
            f'text_split_{stem_algo}',
            mken, tr,
        )


if __name__ == '__main__':
    main()
