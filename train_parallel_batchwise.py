import numpy as np
import math
import random
import os
import multiprocessing as mp

# MNIST data paths
TRAIN_IMAGE = "./data/train-images.idx3-ubyte"
TRAIN_LABEL = "./data/train-labels.idx1-ubyte"
TEST_IMAGE = "./data/t10k-images.idx3-ubyte"
TEST_LABEL = "./data/t10k-labels.idx1-ubyte"

SIZE = 784  # 28*28
NUM_TRAIN = 60000
NUM_TEST = 10000
LEN_INFO_IMAGE = 4
LEN_INFO_LABEL = 2

# Global variables for MNIST data
train_image = np.zeros((NUM_TRAIN, SIZE))
test_image = np.zeros((NUM_TEST, SIZE))
train_label = np.zeros(NUM_TRAIN, dtype=int)
test_label = np.zeros(NUM_TEST, dtype=int)

# Function to flip bytes for MNIST data
def flip_long(ptr):
    val = ptr[0]
    ptr[0] = ptr[3]
    ptr[3] = val
    ptr[1], ptr[2] = ptr[2], ptr[1]

# Function to read MNIST data
def read_mnist_char(file_path, num_data, len_info, arr_n, data_char, info_arr):
    with open(file_path, 'rb') as f:
        info_bytes = f.read(len_info * 4)
        for i in range(len_info):
            ptr = bytearray(info_bytes[i*4:(i+1)*4])
            flip_long(ptr)
            info_arr[i] = int.from_bytes(ptr, byteorder='big')
        for i in range(num_data):
            data_bytes = f.read(arr_n)
            data_char[i] = list(data_bytes)

# Function to convert image data from char to double
def image_char2double(num_data, data_image_char, data_image):
    for i in range(num_data):
        for j in range(SIZE):
            data_image[i][j] = data_image_char[i][j] / 255.0

# Function to convert label data from char to int
def label_char2int(num_data, data_label_char, data_label):
    for i in range(num_data):
        data_label[i] = data_label_char[i][0]

# Function to load MNIST data
def load_mnist():
    global train_image, test_image, train_label, test_label
    train_image_char = np.zeros((NUM_TRAIN, SIZE), dtype=np.uint8)
    test_image_char = np.zeros((NUM_TEST, SIZE), dtype=np.uint8)
    train_label_char = np.zeros((NUM_TRAIN, 1), dtype=np.uint8)
    test_label_char = np.zeros((NUM_TEST, 1), dtype=np.uint8)
    info_image = np.zeros(LEN_INFO_IMAGE, dtype=int)
    info_label = np.zeros(LEN_INFO_LABEL, dtype=int)

    read_mnist_char(TRAIN_IMAGE, NUM_TRAIN, LEN_INFO_IMAGE, SIZE, train_image_char, info_image)
    image_char2double(NUM_TRAIN, train_image_char, train_image)

    read_mnist_char(TEST_IMAGE, NUM_TEST, LEN_INFO_IMAGE, SIZE, test_image_char, info_image)
    image_char2double(NUM_TEST, test_image_char, test_image)

    read_mnist_char(TRAIN_LABEL, NUM_TRAIN, LEN_INFO_LABEL, 1, train_label_char, info_label)
    label_char2int(NUM_TRAIN, train_label_char, train_label)

    read_mnist_char(TEST_LABEL, NUM_TEST, LEN_INFO_LABEL, 1, test_label_char, info_label)
    label_char2int(NUM_TEST, test_label_char, test_label)

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

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Neural Network class
class NeuralNetwork:
    def __init__(self, num_layers, neurons):
        self.num_layers = num_layers
        self.neurons = neurons
        self.weights = [None] * num_layers
        self.bias = [None] * num_layers
        self.z = [None] * num_layers
        self.a = [None] * num_layers
        self.delta = [None] * num_layers
        self.delta_b = [None] * num_layers

        for i in range(1, num_layers):
            self.weights[i] = np.random.randn(neurons[i], neurons[i-1]) * 0.1
            self.bias[i] = np.random.randn(neurons[i]) * 0.1

    def forward(self, batch_image, batch_label, batch_size):
        error = 0.0
        total = 0.0
        for i in range(batch_size):
            image = batch_image[i].copy()
            self.z[0] = image

            for j in range(1, self.num_layers):
                self.z[j] = np.dot(self.weights[j], self.z[j-1]) + self.bias[j]
                self.z[j] = np.array([sigmoid(x) for x in self.z[j]])

            label = np.zeros(self.neurons[self.num_layers - 1])
            label[batch_label[i]] = 1.0

            predicted = np.argmax(self.z[self.num_layers - 1])
            if predicted == batch_label[i]:
                total += 1.0

            self.a[self.num_layers - 1] = self.z[self.num_layers - 1] * (1 - self.z[self.num_layers - 1]) * (label - self.z[self.num_layers - 1])

            image_error = np.sqrt(np.sum((self.z[self.num_layers - 1] - label) ** 2))
            error += image_error

        accuracy = (total / batch_size) * 100
        return error, accuracy

    def backward(self, batch_size, learning_rate):
        for i in range(self.num_layers-2, 0, -1):
            for j in range(batch_size):
                for k in range(self.neurons[i]):
                    error_term = np.sum(self.weights[i+1][:, k] * self.a[i+1])
                    self.a[i][j][k] = self.z[i][j][k] * (1 - self.z[i][j][k]) * error_term

        for i in range(batch_size):
            for j in range(self.num_layers-1):
                for k in range(self.neurons[j]):
                    for l in range(self.neurons[j+1]):
                        self.delta[i][j+1][l][k] = learning_rate * self.a[i][j+1][l] * self.z[i][j][k]

        for i in range(batch_size):
            for j in range(1, self.num_layers):
                for k in range(self.neurons[j]):
                    self.delta_b[i][j][k] = learning_rate * self.a[i][j][k]

        for j in range(self.num_layers-1):
            for k in range(self.neurons[j]):
                for l in range(self.neurons[j+1]):
                    net_delta = np.sum([self.delta[i][j+1][l][k] for i in range(batch_size)])
                    self.weights[j+1][l][k] += net_delta

        for j in range(1, self.num_layers):
            for k in range(self.neurons[j]):
                net_delta = np.sum([self.delta_b[i][j][k] for i in range(batch_size)])
                self.bias[j][k] += net_delta

# Main function
def main():
    import sys
    if len(sys.argv) != 4:
        print("Usage: python train_parallel_batchwise.py [num-epochs] [batch-size] [learning-rate]")
        return

    num_epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    learning_rate = float(sys.argv[3])

    load_mnist()

    print("Num Layers: ", end='')
    num_layers = int(input())

    neurons = []
    for i in range(num_layers):
        print(f"Neurons Layer {i+1}: ", end='')
        neurons.append(int(input()))

    network = NeuralNetwork(num_layers, neurons)

    for epoch in range(num_epochs):
        for i in range(0, NUM_TRAIN, batch_size):
            batch_image = train_image[i:i+batch_size]
            batch_label = train_label[i:i+batch_size]
            error, accuracy = network.forward(batch_image, batch_label, batch_size)
            network.backward(batch_size, learning_rate)
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size + 1}/{NUM_TRAIN//batch_size}, Error: {error:.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main() 