import os
import sys
import time
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# =====================================================================
# PATH CONFIGURATION & CUSTOM IMPORTS
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
logic_dir = os.path.abspath(os.path.join(current_dir, "../../..", "Python-Learning/04_Math_Algorithms/01_Algorithm_Code/01_Matrices_and_Determinants"))
sys.path.append(logic_dir)

try:
    from matrix_operation import Matrix # type: ignore
except ImportError:
    print("Warning: Custom Matrix module not found. Check your directory paths.")

# =====================================================================
# PHASE 1 & 2: SINGLE-CORE EXECUTION (THE BASELINE)
# =====================================================================
def execute_single_core_numpy(inputs_np, weights_np):
    """
    Calculates the forward pass using C-optimized NumPy on a single thread.

    Args:
        inputs_np (numpy.ndarray): The input feature matrix (X).
        weights_np (numpy.ndarray): The weight matrix (W).

    Returns:
        numpy.ndarray: The activated output probabilities after applying the Sigmoid function.
    """
    dot_product_result = np.dot(inputs_np, weights_np)
    activated_output = 1 / (1 + np.exp(-dot_product_result))
    return activated_output


def execute_single_core_custom(inputs_matrix_obj, weights_matrix_obj):
    """
    Calculates the forward pass using the custom pure-Python math engine on a single thread.

    Args:
        inputs_matrix_obj (Matrix): Custom Matrix object containing input features (X).
        weights_matrix_obj (Matrix): Custom Matrix object containing weights (W).

    Returns:
        list of list of float: The activated output probabilities (2D list) after applying 
        the Sigmoid function via pure Python list comprehensions.
    """
    dot_product_result = inputs_matrix_obj.multiply(weights_matrix_obj)
    
    activated_output = [
        [1 / (1 + math.exp(-float(val))) for val in row] 
        for row in dot_product_result.matrix
    ]
    return activated_output


# =====================================================================
# MULTIPROCESSING HELPERS & WORKERS
# =====================================================================
def chunk_matrix(data_matrix, total_chunks):
    """
    Splits a 2D matrix (list or array) into roughly equal row-wise chunks to distribute 
    across multiple CPU cores, minimizing Inter-Process Communication (IPC) overhead.

    Args:
        data_matrix (list or numpy.ndarray): The full input matrix to be divided.
        total_chunks (int): The number of chunks to create (usually matching the CPU core count).

    Returns:
        list: A list containing sub-matrices (chunks) of the original data.
    """
    chunk_size = len(data_matrix) // total_chunks
    matrix_chunks = []
    
    for index in range(total_chunks):
        start_row = index * chunk_size
        # The last chunk absorbs any remaining odd rows to prevent data loss
        end_row = len(data_matrix) if index == total_chunks - 1 else (index + 1) * chunk_size
        matrix_chunks.append(data_matrix[start_row:end_row])
        
    return matrix_chunks


def numpy_worker_task(worker_args):
    """
    Isolated worker function designed to run on a separate CPU core for NumPy operations.

    Args:
        worker_args (tuple): A tuple containing (input_chunk_np, full_weights_np).

    Returns:
        numpy.ndarray: The activated Sigmoid output for the specific chunk.
    """
    input_chunk_np, full_weights_np = worker_args
    chunk_dot_product = np.dot(input_chunk_np, full_weights_np)
    return 1 / (1 + np.exp(-chunk_dot_product))


def custom_engine_worker_task(worker_args):
    """
    Isolated worker function designed to run on a separate CPU core for the Custom Python engine.
    It reconstructs the custom Matrix objects locally to prevent heavy pickling overhead.

    Args:
        worker_args (tuple): A tuple containing (input_chunk_raw_list, full_weights_raw_list).

    Returns:
        list of list of float: The activated Sigmoid output (2D list) for the specific chunk.
    """
    input_chunk_raw, full_weights_raw = worker_args
    
    chunk_rows, chunk_cols = len(input_chunk_raw), len(input_chunk_raw[0])
    weight_rows, weight_cols = len(full_weights_raw), len(full_weights_raw[0])
    
    inputs_mat_obj = Matrix(chunk_rows, chunk_cols, data=input_chunk_raw)
    weights_mat_obj = Matrix(weight_rows, weight_cols, data=full_weights_raw)
    
    chunk_dot_product = Matrix.multiply(inputs_mat_obj, weights_mat_obj) 
    
    return [
        [1 / (1 + math.exp(-float(val))) for val in row] 
        for row in chunk_dot_product.matrix
    ]


# =====================================================================
# MAIN BENCHMARK ORCHESTRATOR
# =====================================================================
def run_full_benchmark():
    """
    Orchestrates the entire execution and benchmarking pipeline. 
    Generates data, configures processes, runs all 4 phases, and prints execution times.
    """
    # Note: Keep ROWS at 1,000,000 for standard testing. 
    TOTAL_ROWS = 1_000_000  
    TOTAL_FEATURES = 50
    HIDDEN_NEURONS = 10
    AVAILABLE_CORES = multiprocessing.cpu_count()

    print("======================================================================")
    print(f"🚀 NEURAL NET ARCHITECTURE BENCHMARK | Cores: {AVAILABLE_CORES} | Rows: {TOTAL_ROWS} 🚀")
    print("======================================================================\n")

    print("[System] Initializing Matrix Data in RAM...")
    
    # Global Data Sourcing
    inputs_np = np.random.rand(TOTAL_ROWS, TOTAL_FEATURES)
    weights_np = np.random.rand(TOTAL_FEATURES, HIDDEN_NEURONS)
    
    inputs_raw_list = inputs_np.tolist()
    weights_raw_list = weights_np.tolist()
    
    inputs_custom_obj = Matrix(TOTAL_ROWS, TOTAL_FEATURES, data=inputs_raw_list)
    weights_custom_obj = Matrix(TOTAL_FEATURES, HIDDEN_NEURONS, data=weights_raw_list)

    # ---------------------------------------------------------
    # PHASE 1 & 2 Execution
    # ---------------------------------------------------------
    start_time = time.perf_counter()
    execute_single_core_numpy(inputs_np, weights_np)
    print(f"Phase 1 (NumPy Single-Core):        {(time.perf_counter() - start_time):.4f} seconds")

    start_time = time.perf_counter()
    execute_single_core_custom(inputs_custom_obj, weights_custom_obj)
    print(f"Phase 2 (Custom Engine Single):     {(time.perf_counter() - start_time):.4f} seconds")

    # ---------------------------------------------------------
    # IPC Preparation
    # ---------------------------------------------------------
    print("\n[System] Slicing matrices for multi-core distribution (IPC Prep)...")
    numpy_chunks = chunk_matrix(inputs_np, AVAILABLE_CORES)
    numpy_worker_payloads = [(chunk, weights_np) for chunk in numpy_chunks]
    
    custom_chunks = chunk_matrix(inputs_raw_list, AVAILABLE_CORES)
    custom_worker_payloads = [(chunk, weights_raw_list) for chunk in custom_chunks]

    # ---------------------------------------------------------
    # PHASE 3 & 4 Execution
    # ---------------------------------------------------------
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=AVAILABLE_CORES) as executor:
        numpy_results = executor.map(numpy_worker_task, numpy_worker_payloads)
    final_numpy_multi_out = np.vstack(list(numpy_results))
    print(f"Phase 3 (NumPy Multi-Core):         {(time.perf_counter() - start_time):.4f} seconds")

    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=AVAILABLE_CORES) as executor:
        custom_results = executor.map(custom_engine_worker_task, custom_worker_payloads)
    final_custom_multi_out = [row for chunk in custom_results for row in chunk]
    print(f"Phase 4 (Custom Engine Multi):      {(time.perf_counter() - start_time):.4f} seconds\n")

if __name__ == '__main__':
    run_full_benchmark()