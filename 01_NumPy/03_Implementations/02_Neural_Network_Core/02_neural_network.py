import os
import sys
import math
import random

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
logic_dir = os.path.abspath(os.path.join(current_dir, "../../..", "Python-Learning/04_Math_Algorithms/01_Algorithm_Code/01_Matrices_and_Determinants"))
sys.path.append(logic_dir)

try:
    from matrix_operation import Matrix # type: ignore
except ImportError:
    print("Warning: Custom Matrix module not found. Check your directory paths.")

# =====================================================================
# NEURAL NETWORK ARCHITECTURE
# =====================================================================
class SimpleNeuralNetwork:
    """
    A lightweight Neural Network built entirely from scratch using a custom Matrix engine.
    
    This class demonstrates the core mechanics of Deep Learning, including 
    forward propagation and backpropagation (gradient descent), without 
    relying on external math libraries like NumPy.
    """

    def __init__(self, input_features, hidden_neurons):
        """
        Initializes the network architecture and generates starting weights.

        Args:
            input_features (int): The number of input data columns (e.g., 3 features).
            hidden_neurons (int): The number of output predictions required.
        """
        self.input_features = input_features
        self.hidden_neurons = hidden_neurons
        
        # Initialize weights with random values between -1.0 and 1.0.
        # This acts as the "blank brain" before the AI starts learning.
        random_weights = [
            [random.uniform(-1.0, 1.0) for _ in range(hidden_neurons)] 
            for _ in range(input_features)
        ]
        
        # Store weights as a custom Matrix object for future dot products
        self.weights = Matrix(input_features, hidden_neurons, data=random_weights)
        
        print(f"[Network] Booted! Architecture: {input_features} Inputs -> {hidden_neurons} Hidden Neurons.")

    def sigmoid_activation(self, Z_matrix):
        """
        Applies the Sigmoid activation function to squash values between 0 and 1.

        Math: f(x) = 1 / (1 + e^-x)

        Args:
            Z_matrix (Matrix): The raw output from the dot product (X . W).

        Returns:
            list: A 2D list of activated probabilities.
        """
        activated_data = [
            [1 / (1 + math.exp(-float(val))) for val in row] 
            for row in Z_matrix.matrix
        ]
        return activated_data

    def forward(self, X_input):
        """
        Executes the Forward Pass: Calculates predictions based on current weights.

        Pipeline: 
        1. Dot Product (Z = X * W)
        2. Activation (Output = Sigmoid(Z))

        Args:
            X_input (Matrix): The input data matrix.

        Returns:
            list: The predicted probabilities.
        """
        # Step 1: Multiply Input data by current Weights
        Z = Matrix.multiply(X_input, self.weights)
        
        # Step 2: Convert raw scores into probabilities (0 to 1)
        output = self.sigmoid_activation(Z)
        
        return output

    def backward(self, X_input, Y_true, output_matrix, learning_rate=0.1):
        """
        Executes the Backward Pass (Backpropagation) to train the network.
        Calculates the error, computes gradients using derivatives, and updates weights.

        Args:
            X_input (Matrix): The original training data.
            Y_true (Matrix): The actual correct answers (Target labels).
            output_matrix (list): The AI's predictions from the forward pass.
            learning_rate (float): How big of a step the AI takes to correct its errors.
        """
        # Step 1: Calculate Error (True Label - AI Prediction)
        error_data = [
            [y_val - out_val for y_val, out_val in zip(y_row, out_row)]
            for y_row, out_row in zip(Y_true.matrix, output_matrix)
        ]
        
        # Step 2: Calculate Gradients (Direction to fix the error)
        # Derivative of Sigmoid = output * (1 - output)
        gradient_data = []
        for i in range(len(output_matrix)):
            row = []
            for j in range(len(output_matrix[0])):
                out_val = output_matrix[i][j]
                err_val = error_data[i][j]
                
                # The slope of the curve at the current prediction
                derivative = out_val * (1.0 - out_val)
                
                # Scale the correction by the learning rate
                row.append(err_val * derivative * learning_rate)
            gradient_data.append(row)
        
        gradient_matrix = Matrix(len(gradient_data), len(gradient_data[0]), data=gradient_data)
        
        # Step 3: Align dimensions using Transpose and calculate Weight Adjustments
        # Delta W = X_Transpose . Gradient
        X_transpose = Matrix.transpose(X_input)
        weight_deltas = Matrix.multiply(X_transpose, gradient_matrix)
        
        # Step 4: Update the network's memory (Weights = Old Weights + Adjustments)
        updated_weights = [
            [w_val + delta_val for w_val, delta_val in zip(w_row, d_row)]
            for w_row, d_row in zip(self.weights.matrix, weight_deltas.matrix)
        ]
        
        # Save the newly learned weights back to the class state
        self.weights = Matrix(len(updated_weights), len(updated_weights[0]), data=updated_weights)

    def train(self, X_input, Y_true, epochs=1000, learning_rate=0.1):
        """
        Trains the neural network by iterating through Forward and Backward passes.

        Args:
            X_input (Matrix): Training features.
            Y_true (Matrix): Target outputs.
            epochs (int): Number of times to loop through the training data.
            learning_rate (float): Speed of learning.
        """
        print(f"[Training] Starting {epochs} Epochs...")
        for epoch in range(epochs):
            # 1. Guess the answers
            output = self.forward(X_input)
            output_matrix = Matrix(len(output), len(output[0]), data=output)
            
            # 2. Learn from mistakes
            self.backward(X_input, Y_true, output, learning_rate)
            
            # Print progress cleanly
            if epoch % 200 == 0:
                print(f"Epoch {epoch} Completed...")
        print("[Training] Network has been trained successfully!\n")

# =====================================================================
# SYSTEM TEST
# =====================================================================
def run_training_test():
    """
    Runs a sandbox test using a simple logical pattern.
    Pattern: The output should strictly match the first column of the input.
    """
    # 1. Setup Dummy Data
    X_raw = [
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ]
    Y_raw = [[0], [1], [1], [0]] # Targets match the first column of X_raw
    
    X_train = Matrix(4, 3, data=X_raw)
    Y_true = Matrix(4, 1, data=Y_raw)

    # 2. Initialize the AI (3 Input Features mapping to 1 Output prediction)
    nn = SimpleNeuralNetwork(input_features=3, hidden_neurons=1)
    
    # 3. Test untrained predictions
    print("\n--- BEFORE TRAINING ---")
    bad_predictions = nn.forward(X_train)
    for res in bad_predictions:
        print(res)

    # 4. Train the model
    print("\n--- STARTING TRAINING ---")
    nn.train(X_train, Y_true, epochs=1000, learning_rate=1.5)

    # 5. Verify final trained predictions
    print("\n--- AFTER TRAINING ---")
    good_predictions = nn.forward(X_train)
    for i, res in enumerate(good_predictions):
        print(f"Target: {Y_raw[i][0]} | AI Prediction: {res[0]:.4f}")

if __name__ == '__main__':
    run_training_test()