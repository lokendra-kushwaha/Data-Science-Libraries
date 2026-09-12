import os
import sys
import time
import matplotlib.image as mpimg

# ==========================================
# 1. ENVIRONMENT & PATH CONFIGURATION
# ==========================================
# Inject custom math algorithms directory into system path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
logic_dir = os.path.abspath(os.path.join(current_dir, "../../../..", "Python-Architecture/04_Math_Algorithms/01_Algorithm_Code/01_Matrices_and_Determinants"))
sys.path.append(logic_dir)

from matrix_operation import Matrix # type: ignore
from image_convolution import apply_convolution # type: ignore
from relu_activation import apply_relu # type: ignore
from max_pooling import apply_max_pooling # type: ignore
from neural_network import SimpleNeuralNetwork
from neural_network_via_numpy import NeuralNetwork

# ==========================================
# 2. IMAGE PIPELINE (VISION SENSOR)
# ==========================================
print("🚀 Loading and preprocessing real image data...")
img_array = mpimg.imread(r"C:\Users\loken\OneDrive\Pictures\919026024270.jpg")

# Convert RGB to Grayscale (Casting to int prevents 8-bit memory overflow)
grayscale_list = []
for i in range(len(img_array)):
    row = []
    for j in range(len(img_array[0])):
        r, g, b = img_array[i][j][:3] 
        gray_value = (int(r) + int(g) + int(b)) / 3.0
        row.append(gray_value)
    grayscale_list.append(row)

image_matrix = Matrix(len(grayscale_list), len(grayscale_list[0]), data=grayscale_list)
print(f"✅ Image loaded into custom Matrix architecture: {image_matrix.row}x{image_matrix.col}")

# Apply CNN Layers: Sobel Edge Detection -> ReLU -> Max Pooling
print("⚙️ Processing CNN Layers (Convolution, Activation, Pooling)...")
sobel_kernel_list = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]
kernel = Matrix(3, 3, data=sobel_kernel_list)

conv_out = apply_convolution(image_matrix, kernel)
relu_out = apply_relu(conv_out)
cnn_final_output = apply_max_pooling(relu_out)

# ==========================================
# 3. THE BRIDGE (FLATTENING LAYER)
# ==========================================
print("\n🔗 Flattening 2D Vision Data for Neural Network ingestion...")

# Professional shortcut: List comprehension to flatten the 2D matrix into 1D
flattened_list = [val for row in cnn_final_output.matrix for val in row]
total_features = len(flattened_list)

print(f"✅ Total extracted AI features: {total_features}")
X_vision_input = Matrix(1, total_features, data=[flattened_list])

# ==========================================
# 4. NEURAL NETWORK (THE BRAIN)
# ==========================================
print(f"\n🧠 Booting AI Brain with {total_features} inputs...")
ai_brain = SimpleNeuralNetwork(input_features=total_features, hidden_neurons=1)

# Baseline Test: Forward Pass with untrained random weights
vision_prediction = ai_brain.forward(X_vision_input)
print(f"🎯 [Untrained State] Raw Prediction Probability: {vision_prediction[0][0]:.4f}")

# ==========================================
# 5. TRAINING & BENCHMARKING
# ==========================================
print("\n🔥 Commencing 10-Epoch Training Benchmark...")
Y_true = Matrix(1, 1, data=[[1.0]])  # Target labeled as 1.0 (True)

start_time = time.time()
ai_brain.train(X_vision_input, Y_true, epochs=10, learning_rate=0.5)
time_taken = time.time() - start_time

print(f"⏱️ Benchmark Complete: 10 Epoch processed in {time_taken:.4f} seconds.")

# Final Inference: Testing weight adjustments post-training
print("\n🧐 Re-evaluating network output post-training...")
new_prediction = ai_brain.forward(X_vision_input)
print(f"🎉 [Trained State] Final Prediction Probability: {new_prediction[0][0]:.4f} (Target: 1.0)")


# ==========================================
# 6. NEURAL NETWORK VIA NUMPY (THE C-ENGINE)
# ==========================================
print(f"\n⚡ Booting NumPy AI Brain with {total_features} inputs...")
ai_brain_np = NeuralNetwork(input_features=total_features, hidden_neurons=1)

# NumPy needs explicit 2D dimensions (Rows x Columns). 
# We wrap flattened_list and 1.0 in extra brackets to make them 1xN and 1x1 matrices.
X_vision_input_np = [flattened_list]
Y_true_np = [[1.0]]

# Baseline Test: Forward Pass with untrained random weights
vision_prediction_np = ai_brain_np.forward(X_vision_input_np)
print(f"🎯 [NumPy Untrained State] Raw Prediction: {vision_prediction_np[0][0]:.4f}")

# ==========================================
# 7. NUMPY TRAINING & BENCHMARKING
# ==========================================
print("\n🚀 Commencing 10-Epoch NumPy Training Benchmark...")

start_time_np = time.time()
ai_brain_np.train(X_vision_input_np, Y_true_np, epochs=10, learning_rate=0.5)
time_taken_np = time.time() - start_time_np

print(f"⏱️ NumPy Benchmark Complete: 10 Epochs processed in {time_taken_np:.6f} seconds.")

# Final Inference: Testing weight adjustments post-training
print("\n🧐 Re-evaluating NumPy network output post-training...")
new_prediction_np = ai_brain_np.forward(X_vision_input_np)
print(f"🎉 [NumPy Trained State] Final Prediction: {new_prediction_np[0][0]:.4f} (Target: 1.0)")

# ==========================================
# 🏆 THE ULTIMATE SPEED SHOWDOWN
# ==========================================
print("\n" + "="*40)
print(" 🏎️  PERFORMANCE SHOWDOWN (10 EPOCHS)")
print("="*40)
print(f"🐍 Pure Python Engine : {time_taken:.6f} Seconds")
print(f"⚙️  NumPy C-Engine     : {time_taken_np:.6f} Seconds")

if time_taken_np > 0:
    speedup = time_taken / time_taken_np
    print(f"🔥 NumPy is roughly {speedup:.2f}x FASTER!")
print("="*40 + "\n")