import numpy as np

class NeuralNetwork:
    """
    A highly optimized, NumPy-accelerated Feed-Forward Neural Network.
    
    This class utilizes vectorized operations and broadcasting to process 
    large datasets significantly faster than pure Python loops.
    """

    def __init__(self, input_features, hidden_neurons):
        """
        Initializes the network architecture with random weights and biases.

        Args:
            input_features (int): Number of features in the input data.
            hidden_neurons (int): Number of neurons in the output/hidden layer.
        """
        self.input_features = input_features
        self.hidden_neurons = hidden_neurons

        # Initialize Weights: Shape (input_features, hidden_neurons)
        # Using np.random.uniform with a direct tuple shape is cleaner
        self.weight = np.random.uniform(-1.0, 1.0, (self.input_features, self.hidden_neurons))
        
        # Initialize Bias: Shape (1, hidden_neurons)
        self.bias = np.random.uniform(-1.0, 1.0, (1, self.hidden_neurons))

    def sigmoid_activation(self, Z):
        """
        Applies the Sigmoid activation safely using NumPy vectorization.
        
        Args:
            Z (np.ndarray): The raw linear combination (X.W + B).
            
        Returns:
            np.ndarray: Probabilities mapped between 0 and 1.
        """
        # NumPy Safety Valve: np.clip restricts values between -700 and 700 
        # in a single line, preventing 'math range error' for massive numbers.
        Z_clipped = np.clip(Z, -700, 700)
        return 1.0 / (1.0 + np.exp(-Z_clipped))

    def forward(self, X_input):
        """
        Performs the forward pass predicting the output.
        """
        # NumPy Broadcasting automatically adds the 1D bias to every row of the 2D dot product
        Z = np.dot(X_input, self.weight) + self.bias
        output = self.sigmoid_activation(Z)
        return output

    def backward(self, X_input, Y_true, output_matrix, learning_rate):
        """
        Calculates gradients and updates weights and biases using Backpropagation.
        """
        # Step 1: Calculate Error and Gradients
        error_data = Y_true - output_matrix
        derivative = output_matrix * (1.0 - output_matrix)
        gradient_data = error_data * derivative * learning_rate

        # Step 2: Update Weights
        X_transpose = X_input.T
        weight_deltas = np.dot(X_transpose, gradient_data)
        self.weight = self.weight + weight_deltas

        # Step 3: Update Biases (Summing gradients vertically across all examples)
        # axis=0 means "summing down the columns", keepdims keeps it as a 2D array
        bias_deltas = np.sum(gradient_data, axis=0, keepdims=True)
        self.bias = self.bias + bias_deltas

    def train(self, X_input, Y_true, epochs=1000, learning_rate=0.1):
        """
        Trains the network over a specified number of epochs.
        """
        X_input = np.array(X_input)
        Y_true = np.array(Y_true)

        for epoch in range(epochs):
            output = self.forward(X_input)
            self.backward(X_input, Y_true, output, learning_rate)


# ==========================================
# 🚀 TESTING THE NUMPY ENGINE
# ==========================================
if __name__ == "__main__":
    X_train = [
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ]   

    Y_true = [[0], [1], [1], [0]]

    print("Initializing NumPy Neural Network...")
    nn = NeuralNetwork(input_features=3, hidden_neurons=1)

    # Ensure inputs are NumPy arrays before passing to forward directly
    X_train_np = np.array(X_train)

    print("\n🎯 Initial Random Predictions:")
    print(nn.forward(X_train_np))

    print("\n⏳ Training for 1000 Epochs...")
    nn.train(X_train, Y_true, epochs=1000, learning_rate=1.5)

    print("\n✅ Final Predictions After Learning:")
    print(nn.forward(X_train_np))