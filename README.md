# Parallel Spell Checking with MPI and CUDA

Spell checker built around three edit distance algorithms, parallelised two ways with MPI
(text-split and dictionary-split) and additionally ported to CUDA. Tested on Macedonian,
English and Turkish.

## Algorithms

| Algorithm | Notes |
|---|---|
| Levenshtein | classic DP, insert / delete / substitute |
| Damerau-Levenshtein | adds transposition as one operation |
| Myers bit-vector | bit-parallel, packs one DP row into machine words |

## Parallelisation strategies

**Text-split.** The list of misspelled words is divided among the processes. Each process
keeps the whole dictionary and handles its own slice of words. Communication happens once
at scatter and once at gather, so the number of MPI operations does not grow with the
input size.

**Dictionary-split.** The dictionary is divided instead. Every process scores its part of
the candidates for the current word, and the partial best results are reduced. This means
a reduce per word, so N words cost O(N) collective operations.

## Requirements

- Python 3.8+
- MPI runtime: MS-MPI on Windows, OpenMPI on Linux
- mpi4py
- CUDA Toolkit and Numba, only if you want the GPU part

```bash
git clone https://github.com/sanevat/parallel-spell-checker.git
cd parallel-spell-checker
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Usage

```python
from spell_checker.algorithms import levenshtein, damerau_levenshtein, myers_bitvector

levenshtein("kitten", "sitting")          # 3
myers_bitvector("kitten", "sitting")      # 3, but faster
damerau_levenshtein("ab", "ba")           # 1, transposition counts once
```

MPI benchmarks:

```bash
mpiexec -n 4 python scripts/benchmarks_mpi/mpi_text_split.py
mpiexec -n 4 python scripts/benchmarks_mpi/mpi_dict_split_levenshtein.py
mpiexec -n 4 python scripts/benchmarks_mpi/mpi_text_split_turkish_myers.py
```

GPU benchmarks:

```bash
python scripts/cuda_vs_c_benchmark_all.py
python scripts/cuda_batch_all.py
```

## Results

All timings below are for 5000 misspelled words. Dictionaries are the Hunspell word lists
(around 26k candidates per lookup for MK/EN, 20k for TR after filtering).

### Sequential baseline

| Algorithm | Lang | Time (s) | ms/word | Accuracy |
|---|---|---|---|---|
| Levenshtein | MK | 1613 | 323 | 79.3% |
| Levenshtein | EN | 1455 | 291 | 79.0% |
| Levenshtein | TR | 1575 | 315 | 77.4% |
| Damerau-Levenshtein | MK | 2409 | 482 | 87.1% |
| Damerau-Levenshtein | EN | 2215 | 443 | 86.8% |
| Damerau-Levenshtein | TR | 2462 | 493 | 85.7% |
| Myers bit-vector | MK | 507 | 102 | 79.3% |
| Myers bit-vector | EN | 418 | 84 | 79.0% |
| Myers bit-vector | TR | 398 | 80 | 90.1% |

Myers returns the same distances as Levenshtein, so the accuracy is identical on MK and EN.
The TR difference comes from the Turkish run using a different candidate set.

### MPI scaling, Levenshtein at 5000 words

Text-split:

| Lang | S(2) | S(4) | S(8) | E(8) |
|---|---|---|---|---|
| MK | 1.94x | 3.22x | 4.86x | 61% |
| EN | 1.96x | 2.92x | 4.74x | 59% |
| TR | 1.69x | 2.61x | 3.26x | 41% |

Dictionary-split, same workload:

| Lang | S(2) | S(4) | S(8) | E(8) |
|---|---|---|---|---|
| MK | 1.76x | 3.08x | 3.93x | 49% |
| EN | 1.77x | 3.04x | 3.91x | 49% |
| TR | 1.58x | 2.21x | 2.63x | 33% |

Text-split wins across the board, and the gap widens with the process count. Full tables for
every algorithm, language and word count are in `results/text_split/` and `results/dict_split/`.

### CUDA vs single-threaded C

Measured on an RTX 2050, median over 24 configurations (3 languages x 8 sizes, 4 runs each):

| Algorithm | Global memory | Shared memory |
|---|---|---|
| Levenshtein | 8.6x | 10.9x |
| Damerau-Levenshtein | 19.8x | 26.5x |
| Myers bit-vector | 11.1x | 19.2x |

Raw numbers in `results/cuda_shared_vs_c.json`.

## What the numbers say

Myers is roughly 3x faster than plain Levenshtein with no loss in accuracy, which makes it
the sensible default. Damerau-Levenshtein buys about 8 percentage points of accuracy on MK
and EN because most real typos are transpositions, but it costs around 50% more time.

Scaling stays close to linear up to 4 processes and then degrades. At 8 processes the runs
were oversubscribing the physical cores, which explains most of the drop. Turkish scales
worse than the other two languages because its dictionary is smaller, so there is less work
per process to hide the communication.

## Layout

```
spell_checker/          algorithms, sequential and CUDA implementations
scripts/benchmarks_mpi/ MPI benchmarks, text-split and dict-split
scripts/plotting/       figure generation
scripts/generate_test_data/ typo injection and ground truth
data/                   dictionaries, test texts, ground truth
results/                raw benchmark output
visualizations/         generated plots
```

## Reference

Myers, G. (1999). A fast bit-vector algorithm for approximate string matching based on
dynamic programming. *Journal of the ACM*, 46(3), 395-415.

## License

MIT

## Author

Teodora Saneva
