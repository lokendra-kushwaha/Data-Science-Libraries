# Custom Neural Network Core

A from-scratch, pure Python implementation of a Machine Learning engine. This module acts as the "AI Brain" of the project, built to deeply understand the underlying mechanics of Artificial Intelligence—without relying on high-level libraries like PyTorch or TensorFlow.

## 🚀 Objective
To demystify "AI" by building the mathematical foundation (Forward Propagation, Backpropagation, Gradient Descent) using our custom Matrix engine. This proves that neural networks are simply a structured pipeline of matrix multiplications and calculus.

## 📁 Local Directory Structure

    03_Implementations/
    │
    ├── 01_Engine_Benchmarks/           # Hardware & Multiprocessing Stress Tests
    │   ├── 01_engine_benchmark.py
    │   └── benchmark_report.md
    │
    └── 02_Neural_Network_Core/         # The AI Brain (Current Directory)
        ├── 02_neural_network.py        # Forward/Backward Pass & Training Loop
        └── README.md                   # This documentation


## 🧠 Core Features of `02_neural_network.py`
* **Forward Propagation:** Computes predictions using custom matrix multiplication and Sigmoid activation.
* **Backpropagation:** Implements the Chain Rule, error calculation, and weight updates via Gradient Descent.
* **Zero Dependencies:** 100% pure Python. Imports its logic strictly from our custom `matrix_operation.py` backend.

## ⚙️ How The Training Loop Works
The network learns by iteratively adjusting its random weights through four mathematical steps:
1. **Error Calculation:** Subtracts the AI's prediction from the true label.
2. **Gradient Calculation:** Uses the derivative of the Sigmoid function to find the slope of correction.
3. **Matrix Transpose:** Transposes the input matrix to align dimensions for the weight adjustment dot product.
4. **Weight Update:** Applies the calculated deltas (scaled by a learning rate) to the network's memory.

## 💻 Sample Output (Before vs. After Training)

    [Network] Booted! Architecture: 3 Inputs -> 1 Hidden Neurons.

    --- BEFORE TRAINING ---
    [0.6561]  # Random garbage predictions
    [0.3731]
    [0.5164]
    [0.5154]

    --- STARTING TRAINING ---
    [Training] Starting 1000 Epochs...
    Epoch 0 Completed...
    ...
    [Training] Network has been trained successfully!

    --- AFTER TRAINING ---
    Target: 0 | AI Prediction: 0.0258  # Successfully learned the pattern!
    Target: 1 | AI Prediction: 0.9790
    Target: 1 | AI Prediction: 0.9831
    Target: 0 | AI Prediction: 0.0208