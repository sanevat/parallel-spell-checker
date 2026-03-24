"""
Combined Visualization for Paper

Creates a single figure with all 5 scalability visualizations arranged
in a grid layout suitable for academic papers.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Research paper style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

results_dir = Path(__file__).parent
viz_dir = results_dir / 'visualizations' / 'new_res'

# Define the 5 visualizations with their labels
visualizations = [
    ('levenshtein_scalability_new.png', '(a) Text-Split: Levenshtein'),
    ('damerau_levenshtein_scalability_new.png', '(b) Text-Split: Damerau-Levenshtein'),
    ('myers_bitvector_scalability_new.png', '(c) Text-Split: Myers Bit-Vector'),
    ('levenshtein_dictsplit_scalability.png', '(d) Dict-Split: Levenshtein'),
    ('damerau_levenshtein_dictsplit_scalability.png', '(e) Dict-Split: Damerau-Levenshtein'),
]

# Create figure with 3 rows, 2 columns layout
# Row 1: Text-Split Levenshtein, Text-Split Damerau-Levenshtein
# Row 2: Text-Split Myers, Dict-Split Levenshtein
# Row 3: Dict-Split Damerau-Levenshtein (centered)

fig = plt.figure(figsize=(16, 20))

# Grid positions for 5 plots in a 3x2 arrangement
positions = [
    (3, 2, 1),  # Row 1, Col 1
    (3, 2, 2),  # Row 1, Col 2
    (3, 2, 3),  # Row 2, Col 1
    (3, 2, 4),  # Row 2, Col 2
    (3, 2, 5),  # Row 3, Col 1 (will span to center)
]

for i, (filename, label) in enumerate(visualizations):
    img_path = viz_dir / filename
    if not img_path.exists():
        print(f"Warning: {img_path} not found")
        continue

    img = mpimg.imread(str(img_path))

    if i < 4:
        ax = fig.add_subplot(positions[i][0], positions[i][1], positions[i][2])
    else:
        # Center the 5th plot by using a different approach
        ax = fig.add_subplot(3, 2, (5, 6))

    ax.imshow(img)
    ax.set_title(label, fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')

# Adjust layout
plt.tight_layout(pad=2.0)

# Save the combined figure
output_dir = results_dir / 'visualizations' / 'paper'
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / 'combined_scalability_all.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(output_dir / 'combined_scalability_all.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"Saved combined visualization:")
print(f"  - {output_dir / 'combined_scalability_all.png'}")
print(f"  - {output_dir / 'combined_scalability_all.pdf'}")

plt.close()

# Also create an alternative layout: 2 rows for Text-Split (3 plots) and 1 row for Dict-Split (2 plots)
fig2, axes = plt.subplots(3, 2, figsize=(16, 18))

# Flatten and handle the 5 plots
plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]

for i, (filename, label) in enumerate(visualizations):
    img_path = viz_dir / filename
    if not img_path.exists():
        continue

    img = mpimg.imread(str(img_path))
    row, col = plot_positions[i]
    axes[row, col].imshow(img)
    axes[row, col].set_title(label, fontsize=14, fontweight='bold', pad=10)
    axes[row, col].axis('off')

# Hide the empty subplot (row 2, col 1 - bottom right)
axes[2, 1].axis('off')

# Add section labels
fig2.text(0.5, 0.98, 'MPI Scalability Analysis', ha='center', va='top',
          fontsize=18, fontweight='bold')
fig2.text(0.5, 0.67, 'Text-Split Strategy', ha='center', va='top',
          fontsize=14, fontstyle='italic', color='gray')
fig2.text(0.5, 0.34, 'Dict-Split Strategy', ha='center', va='top',
          fontsize=14, fontstyle='italic', color='gray')

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0)

plt.savefig(output_dir / 'combined_scalability_labeled.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(output_dir / 'combined_scalability_labeled.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"  - {output_dir / 'combined_scalability_labeled.png'}")
print(f"  - {output_dir / 'combined_scalability_labeled.pdf'}")

plt.close()
print("\nDone!")
