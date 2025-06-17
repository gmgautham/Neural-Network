import subprocess
import sys


# Artificial inputs for training
num_epochs = 10
batch_size = 32
learning_rate = 0.01
num_layers = 3
neurons_per_layer = [784, 128, 10]  # Example: 784 input neurons, 128 hidden neurons, 10 output neurons

# Simulate user input for train.py
def run_training():
    process = subprocess.Popen(
        [sys.executable, 'train.py', str(num_epochs), str(batch_size), str(learning_rate)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Simulate user input for number of layers and neurons
    process.stdin.write(f"{num_layers}\n")
    for neurons in neurons_per_layer:
        process.stdin.write(f"{neurons}\n")
    process.stdin.flush()

    # Capture output
    stdout, stderr = process.communicate()
    print("Output:", stdout)
    if stderr:
        print("Errors:", stderr)

if __name__ == "__main__":
    run_training() 