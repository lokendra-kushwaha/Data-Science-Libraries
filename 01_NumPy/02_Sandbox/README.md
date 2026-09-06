# ⚙️ NumPy System Architecture Sandbox

Welcome to the high-performance testing ground for NumPy. This sandbox is dedicated to understanding how NumPy interacts with physical hardware—moving beyond standard API syntax to explore memory pointers, CPU cache mechanics, and C-engine optimizations.

## 📂 Core Focus Areas

All benchmarking and testing scripts in this directory are categorized by the following underlying system mechanics:

* **Memory Layouts & CPU Caching:** Benchmarking C-Order (Row-major) vs. F-Order (Column-major), spatial locality, and Stride mathematics.
* **Pointers (Views vs. Copies):** Hardware-level proofs of memory sharing across standard slicing, boolean masking, and fancy indexing.
* **Engine Execution:** Testing Vectorization limits, BLAS matrix multiplication engines, and SIMD parallel processing.
* **Safety & Filtration:** Handling silent memory overflow traps (`astype` downcasting) and managing dirty data via `numpy.ma` (Masked Arrays) without breaking physical strides.
* **Zero-Copy Engine:** Advanced N-dimensional iteration (`np.nditer`) and in-place memory mutation using `out=` parameters and `x[...]` buffer pointers to prevent RAM spikes.

## 📖 The Master Reference

For detailed theoretical teardowns, exact hardware rules, and the "why" behind the code in this sandbox, refer to the core architecture guide:
**➔ [`NumPy_Architecture_Guide.md`](./NumPy_Architecture_Guide.md)**

## 🚀 How to Run

Execute any script directly via the terminal to observe live RAM footprint optimization, boolean pointer proofs (`np.shares_memory`), and CPU execution times measured down to the microsecond.