import numpy as np

# Activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron training function
def perceptron_train(X, y, learning_rate=0.1, epochs=10):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        for idx, x_i in enumerate(X):
            # Linear combination
            linear_output = np.dot(x_i, weights) + bias
            # Apply activation function
            y_pred = step_function(linear_output)

            # Update weights and bias
            update = learning_rate * (y[idx] - y_pred)
            weights += update * x_i
            bias += update

    return weights, bias

# Perceptron prediction function
def perceptron_predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return np.array([step_function(x) for x in linear_output])

# Training data
# AND Gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # Output for AND gate

# OR Gate
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])  # Output for OR gate

# XOR Gate (Perceptron will fail for this)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # Output for XOR gate

# Train perceptron for AND
weights_and, bias_and = perceptron_train(X_and, y_and)
predictions_and = perceptron_predict(X_and, weights_and, bias_and)

# Train perceptron for OR
weights_or, bias_or = perceptron_train(X_or, y_or)
predictions_or = perceptron_predict(X_or, weights_or, bias_or)

# Train perceptron for XOR
weights_xor, bias_xor = perceptron_train(X_xor, y_xor)
predictions_xor = perceptron_predict(X_xor, weights_xor, bias_xor)

# Print results
print("AND Gate")
print(f"Predictions: {predictions_and}")

print("\nOR Gate")
print(f"Predictions: {predictions_or}")

print("\nXOR Gate (Perceptron cannot solve this correctly)")
print(f"Predictions: {predictions_xor}")
