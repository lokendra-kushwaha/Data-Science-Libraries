# Custom Matrix Engine Benchmarks

This module contains stress tests and performance benchmarks for our custom, pure-Python Matrix engine. The goal is to analyze hardware limitations, Memory (RAM) saturation, and CPU execution times without relying on optimized C-based libraries like NumPy.

## 🚀 Objective
To test how pure Python lists handle massive datasets (up to 1 Million rows) and to evaluate if Python's `multiprocessing` module genuinely speeds up matrix math or if it introduces IPC (Inter-Process Communication) bottlenecks.

## 📁 Local Directory Structure

    03_Implementations/
    │
    ├── 01_Engine_Benchmarks/           # Benchmarks (Current Directory)
    │   ├── 01_engine_benchmark.py      # The main stress-testing script
    │   ├── benchmark_report.md         # Detailed findings from the 1M rows test
    │   └── README.md                   # This documentation
    │
    └── 02_Neural_Network_Core/         # The AI Brain
        └── ...


## 🧠 Core Learnings & Insights
* **The IPC Bottleneck:** Multiprocessing is not a silver bullet in Python. For standard matrix math, single-core processing often beats multi-core due to the overhead of "pickling" (serializing/deserializing) massive data chunks across CPU cores.
* **RAM Saturation:** Pure Python lists are highly memory-inefficient. Stress testing with 1 Million rows proved that data must be processed in chunks, which directly translates to the concept of **"Mini-Batching"** and **DataLoaders** in real-world AI training.
* **Architectural Validation:** This test confirmed why libraries like PyTorch/TensorFlow use C++ and GPUs for parallel processing, and set the hardware baseline for our custom Neural Network engine.

## ⚙️ How to Run
Execute the benchmark script directly to test your own hardware limits. 

*Note: Proceed with caution. Testing with 1 Million rows will consume significant RAM and may temporarily freeze lower-end systems.*