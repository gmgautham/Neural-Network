import numpy as np
import math
import random

# Data for OR gate
features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# labels = np.array([0, 1, 1, 1])  # labels for OR gate
labels = np.array([0, 0, 0, 1])  # labels for AND gate

# Function to generate Gaussian random numbers
def gaussrand():
    if not hasattr(gaussrand, 'phase'):
        gaussrand.phase = 0
        gaussrand.V1 = 0
        gaussrand.V2 = 0
        gaussrand.S = 0

    if gaussrand.phase == 0:
        while True:
            U1 = random.random()
            U2 = random.random()
            gaussrand.V1 = 2 * U1 - 1
            gaussrand.V2 = 2 * U2 - 1
            gaussrand.S = gaussrand.V1 * gaussrand.V1 + gaussrand.V2 * gaussrand.V2
            if gaussrand.S < 1 and gaussrand.S != 0:
                break
        X = gaussrand.V1 * math.sqrt(-2 * math.log(gaussrand.S) / gaussrand.S)
    else:
        X = gaussrand.V2 * math.sqrt(-2 * math.log(gaussrand.S) / gaussrand.S)
    gaussrand.phase = 1 - gaussrand.phase
    return X

# Signum function
def signum(n):
    return 1.0 if n > 0.0 else -1.0

# Adaline class
class Adaline:
    def __init__(self, num_weights):
        self.weights = np.ones(num_weights) * 0.1
        self.bias = 0.1
        self.delta_weights = np.zeros(num_weights)
        self.delta_bias = 0.0

    def forward(self, batch_feature, batch_label, batch_size, learning_rate):
        error = 0.0
        self.delta_weights.fill(0.0)
        self.delta_bias = 0.0

        for i in range(batch_size):
            curr_hyp = np.dot(self.weights, batch_feature[i]) + self.bias
            curr_error = (signum(curr_hyp) - signum(batch_label[i])) ** 2 / 2.0
            error += curr_error

            self.delta_weights += learning_rate * (signum(curr_hyp) - signum(batch_label[i])) * batch_feature[i]
            self.delta_bias += learning_rate * (signum(curr_hyp) - signum(batch_label[i]))

        return error

    def backward(self):
        self.weights += self.delta_weights
        self.bias += self.delta_bias
        print("New weights:", self.weights, "New Bias:", self.bias)

# Main function
def main():
    import sys
    if len(sys.argv) != 5:
        print("Usage: python adaline.py [num-epochs] [batch-size] [learning-rate] [num-weights]")
        return

    num_epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    num_weights = int(sys.argv[4])

    adaline = Adaline(num_weights)

    for i in range(num_epochs):
        num_batches = 4 // batch_size  # 4 is dataset size

        for j in range(num_batches):
            batch_feature = features[batch_size*j:batch_size*(j+1)]
            batch_label = labels[batch_size*j:batch_size*(j+1)]

            error = adaline.forward(batch_feature, batch_label, batch_size, learning_rate)
            adaline.backward()
            print(f"Epoch [{i+1}/{num_epochs}], Batch [{j+1}/{num_batches}], Error: {error}")

if __name__ == "__main__":
    main() 