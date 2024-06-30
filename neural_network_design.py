import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from ocr import OCRNeuralNetwork
from tqdm import tqdm

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize images to the range [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Flatten the images to 1D arrays of 784 pixels
train_images = train_images.reshape((train_images.shape[0], 28 * 28))
test_images = test_images.reshape((test_images.shape[0], 28 * 28))

# Assume data_matrix, data_labels, train_indices, and test_indices are defined.
# They should be loaded or set up as needed for training/testing.

def test(data_matrix, data_labels, test_indices, nn):
    avg_accuracy = 0
    for j in range(100):  # 100 iterations for averaging
        correct_guess_count = 0
        for i in test_indices:
            test = data_matrix[i]
            prediction = nn.predict(test)
            if data_labels[i] == prediction:
                correct_guess_count += 1
        avg_accuracy += (correct_guess_count / float(len(test_indices)))
    return avg_accuracy / 100

# Evaluate performance for different hidden node configurations
for num_hidden_nodes in tqdm(range(5, 50, 5)):
    nn = OCRNeuralNetwork(num_hidden_nodes=num_hidden_nodes, use_file=False)
    nn.train(training_data=[{'y0': train_images[i], 'label': train_labels[i]} for i in range(len(train_images))])
    performance = test(test_images, test_labels, range(len(test_images)), nn)
    print(f"{num_hidden_nodes} Hidden Nodes: {performance:.4f}")
