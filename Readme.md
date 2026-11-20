# MLScratch

Building a machine learning library from scratch in C++. Right now, the focus is on getting the foundational math and data pipeline right before moving on to building actual models on top of it. 

I've put together a custom matrix engine to handle core linear algebra operations like matrix inversion and determinants. It's written in C++20, manages memory safely with the Rule of Five, and uses Google Benchmark to keep track of how the algorithms scale as the matrices get larger.

Here is a snapshot of the current performance scaling for inversions and determinants:

![Benchmark Results](imgs/benchmark_mat.png)

## What's Built So Far

With the core math engine in place, the rest of the data and evaluation pipeline is now ready:

* **Matrix Engine (`utils`):** Handles the core linear algebra—cache-tiled multiplication, transpose, Hadamard products, broadcasting, axis reductions (`sum`, `mean`, `min`, `max`), inversion, determinants, and random initialization.
* **CSV Loader:** Reads raw datasets from disk directly into matrices and splits them into features (`X`) and targets (`y`).
* **Preprocessing (`preprocess`):** Includes a reproducible `train_test_split` (shuffling lightweight row indices instead of swapping matrix memory) along with `StandardScaler` and `MinMaxScaler` using the `fit`/`transform` pattern to prevent data leakage.
* **Evaluation Metrics (`metrics`):** Basic scoring functions to evaluate model performance, starting with Mean Squared Error (MSE) and Classification Accuracy.

Next up is building out the first set of classical ML models on top of this pipeline.

## Building and Running

The project uses standard CMake. To compile and run the benchmarks yourself, it's best to build in Release mode so the compiler optimizations kick in:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Run the benchmarks
./build/Release/ml_benchmark