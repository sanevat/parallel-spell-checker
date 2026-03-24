# Parallel Spell Checking with MPI

A high-performance spell checking system using parallel computing with MPI. Implements multiple edit distance algorithms and compares two parallelization strategies: **text-split** and **dictionary-split**.

## Features

- **Three Edit Distance Algorithms:**
  - **Levenshtein** - Standard edit distance (insert, delete, substitute)
  - **Damerau-Levenshtein** - Includes transposition as single operation (+7-9% accuracy)
  - **Myers' Bit-Vector** - Bit-parallel algorithm (3-5x faster than Levenshtein)

- **Two Parallelization Strategies:**
  - **Text-Split** - Distributes misspelled words across processes (better efficiency)
  - **Dictionary-Split** - Distributes dictionary candidates across processes

- **Multi-language Support:**
  - Macedonian (Cyrillic/UTF-8)
  - English (ASCII)

## Project Structure

```
spell_checking/
├── spell_checker/
│   ├── algorithms/
│   │   ├── levenshtein.py
│   │   ├── damerau_levenshtein.py
│   │   └── myers_bitvector.py
│   ├── spell_checker.py
│   └── benchmark.py
├── scripts/
│   ├── benchmarks_mpi/
│   │   ├── mpi_text_split.py
│   │   ├── mpi_dict_split_levenshtein.py
│   │   └── ...
│   └── plotting/
├── data/
│   ├── dictionary/
│   ├── ground_truth/
│   └── test_texts/
├── results/
│   ├── text_split/
│   └── dict_split/
└── visualizations/
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/spell_checking.git
cd spell_checking

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- MPI implementation (MS-MPI for Windows, OpenMPI for Linux)
- mpi4py

## Usage

### Basic Spell Checking

```python
from spell_checker.algorithms import levenshtein, damerau_levenshtein, myers_bitvector

# Calculate edit distance
dist = levenshtein("kitten", "sitting")  # Returns 3
dist = myers_bitvector("kitten", "sitting")  # Returns 3 (faster)
dist = damerau_levenshtein("ab", "ba")  # Returns 1 (transposition)
```

### Running MPI Benchmarks

```bash
# Text-split strategy with 4 processes
mpiexec -n 4 python scripts/benchmarks_mpi/mpi_text_split.py

# Dictionary-split strategy with 4 processes
mpiexec -n 4 python scripts/benchmarks_mpi/mpi_dict_split_levenshtein.py
```

## Performance Results

### Algorithm Comparison (Sequential, 5000 words)

| Algorithm | Time (s) | ms/word | Accuracy |
|-----------|----------|---------|----------|
| Levenshtein | 1523 | 305 | 79.1% |
| Damerau-Levenshtein | 2314 | 463 | 86.9% |
| Myers Bit-Vector | 450 | 90 | 79.1% |

### Parallel Speedup (Text-Split, 5000 words)

| Processes | Speedup | Efficiency |
|-----------|---------|------------|
| 1 | 1.00x | 100% |
| 2 | 1.97x | 98% |
| 4 | 3.60x | 90% |
| 8 | 5.00x | 62% |

### Text-Split vs Dictionary-Split

| Strategy | MPI Ops (N words) | Efficiency (8P) |
|----------|-------------------|-----------------|
| Text-Split | 5 | 60-63% |
| Dict-Split | 5N | 40-49% |

## Key Findings

1. **Myers' bit-vector** achieves 3.2x speedup over Levenshtein with identical accuracy
2. **Damerau-Levenshtein** provides ~8% higher accuracy by handling transpositions
3. **Text-split** parallelization is superior with 1000x fewer MPI operations
4. Near-linear scaling up to 4 processes (>90% efficiency)

## Visualizations

Generated plots are saved in `visualizations/`:
- Scalability curves for each algorithm
- Speedup and efficiency analysis
- Strategy comparison charts

## License

MIT License

## Author

[Your Name]

## Acknowledgments

- Myers' bit-vector algorithm based on: Myers, G. (1999). "A fast bit-vector algorithm for approximate string matching based on dynamic programming"
