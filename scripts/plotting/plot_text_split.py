import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
    ('results/text_split_250.json', 250),
    ('results/text_split_500.json', 500),
    ('results/text_split_1K.json', 1000),
    ('results/text_split_1500.json', 1500),
    ('results/text_split_2K.json', 2000),
    ('results/text_split_2500.json', 2500),
    ('results/text_split_3500.json', 3500),
    ('results/text_split_5000.json', 5000),
]

# Data structure: {num_procs: {language: [times for each word count]}}
data = {2: {'MK': [], 'EN': []}, 4: {'MK': [], 'EN': []}, 8: {'MK': [], 'EN': []}}
word_counts = []

for file_path, num_words in file_info:
    word_counts.append(num_words)
    with open(file_path, 'r') as f:
        results = json.load(f)

    text_split = results.get('text_split', {})

    for num_procs in ['2', '4', '8']:
        if num_procs in text_split:
            for entry in text_split[num_procs]:
                if entry['algorithm'] == 'Myers Bit-Vector':
                    lang = entry['language']
                    if 'total_time_s' in entry:
                        time_s = entry['total_time_s']
                    else:
                        time_s = entry['ms_per_word'] * num_words / 1000
                    data[int(num_procs)][lang].append(time_s)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5.5))

# Colors and markers for different process counts
colors = {2: '#D62728', 4: '#2CA02C', 8: '#FF7F0E'}  # red, green, orange
markers_mk = {2: 'o', 4: 's', 8: '^'}  # circle, square, triangle for MK
markers_en = {2: 'o', 4: 's', 8: '^'}  # same markers for EN

# Plot MK (solid lines) first
for num_procs in [2, 4, 8]:
    times = data[num_procs]['MK']
    if len(times) == len(word_counts):
        ax.plot(word_counts, times,
               color=colors[num_procs],
               linestyle='-',
               marker=markers_mk[num_procs],
               markerfacecolor=colors[num_procs],
               markeredgecolor='black',
               markeredgewidth=0.5,
               label=f'{num_procs} Processes (MK)')

# Plot EN (dashed lines)
for num_procs in [2, 4, 8]:
    times = data[num_procs]['EN']
    if len(times) == len(word_counts):
        ax.plot(word_counts, times,
               color=colors[num_procs],
               linestyle='--',
               marker=markers_en[num_procs],
               markerfacecolor='white',
               markeredgecolor=colors[num_procs],
               markeredgewidth=1.5,
               label=f'{num_procs} Processes (EN)')

# Configure the plot
ax.set_xlabel('Number of Words')
ax.set_ylabel('Time (seconds)')
ax.set_title('MPI Text-Split Scalability: Myers Bit-Vector')

# Set x-axis ticks
ax.set_xticks(word_counts)
ax.set_xticklabels(word_counts)

# Create custom legend with clear distinction
from matplotlib.lines import Line2D

legend_elements = [
    # Header-like entries for line styles
    Line2D([0], [0], color='black', linestyle='-', marker='o',
           markerfacecolor='gray', markeredgecolor='black', markeredgewidth=0.5,
           label='MK (solid, filled)'),
    Line2D([0], [0], color='black', linestyle='--', marker='o',
           markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5,
           label='EN (dashed, hollow)'),
    Line2D([0], [0], color='none', label=''),  # spacer
    # Process count entries
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
ax.set_xlim(0, 5250)
ax.set_ylim(0, None)

# Tight layout
plt.tight_layout()

# Save the figure
plt.savefig('text_split_scalability_myers_bitvector.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('text_split_scalability_myers_bitvector.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Graph saved as 'text_split_scalability_myers_bitvector.png' and '.pdf'")
