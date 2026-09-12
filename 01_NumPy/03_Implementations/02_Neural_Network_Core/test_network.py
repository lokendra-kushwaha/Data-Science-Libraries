import os
import sys

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
logic_dir = os.path.abspath(os.path.join(current_dir, "../../../..", "Python-Architecture/04_Math_Algorithms/01_Algorithm_Code/01_Matrices_and_Determinants"))
sys.path.append(logic_dir)
print(logic_dir)
try:
    from matrix_operation import Matrix # type: ignore
except ImportError:
    print("Warning: Custom Matrix module not found. Check your directory paths.")

from neural_network import SimpleNeuralNetwork

print("🚀 Initiating Neural Network Test (OR Gate Logic)...")

# ==========================================
# 1. THE DATASET
# ==========================================
X_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y_data = [
    [0],
    [1],
    [1],
    [1]
]

X_input = Matrix(4, 2, data=X_data)
Y_true = Matrix(4, 1, data=Y_data)

# ==========================================
# 2. BOOTING THE AI BRAIN 
# ==========================================

ai_brain = SimpleNeuralNetwork(input_features=2, hidden_neurons=1)

print("\n🧠 [Initial State] AI's random blind guesses before learning:")
initial_guess = ai_brain.forward(X_input)
for i in range(4):
    print(f"Input {X_data[i]} -> AI Guess: {initial_guess[i][0]:.4f} | Target: {Y_data[i][0]}")

# ==========================================
# 3. THE TRAINING PHASE
# ==========================================
print("\n⏳ Starting the Engine... (Running Forward & Backward passes)")

ai_brain.train(X_input, Y_true, epochs=2000, learning_rate=0.5)

# ==========================================
# 4. THE FINAL RESULT
# ==========================================
print("🎓 [Final Result] AI Predictions After Learning:")
final_predictions = ai_brain.forward(X_input)

for i in range(4):
    pred = final_predictions[i][0]
    actual = Y_data[i][0]
    print(f"Input: {X_data[i]} | AI Guessed: {pred:.4f} (Target: {actual})")