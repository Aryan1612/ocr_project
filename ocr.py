import numpy as np
import json
import os

class OCRNeuralNetwork:
    NN_FILE_PATH = 'nn.json'
    LEARNING_RATE = 0.1
    
    def __init__(self, num_hidden_nodes = 35, input_size=784, output_size=10, use_file=True):
        self.num_hidden_nodes = num_hidden_nodes
        self.input_size = input_size
        self.output_size = output_size
        self._use_file = use_file
        
        # Initialize weights and biases
        self.theta1 = self._rand_initialize_weights(num_hidden_nodes, input_size)
        self.theta2 = self._rand_initialize_weights(output_size, num_hidden_nodes)
        self.input_layer_bias = self._rand_initialize_weights(num_hidden_nodes, 1)  # Column vector
        self.hidden_layer_bias = self._rand_initialize_weights(output_size, 1)  # Column vector
        
        if self._use_file:
            self._load()
    
    def _rand_initialize_weights(self, size_out, size_in):
        return np.random.randn(size_out, size_in) * 0.1
    
    def _sigmoid(self, z):
        z_clipped = np.clip(z, -500, 500)  # Clip values of z to prevent overflow
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    def _sigmoid_prime(self, z):
        sigmoid_z = self._sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)
    
    def train(self, training_data):
        for data in training_data:
            # Feedforward
            y0 = np.array(data['y0']).reshape(-1, 1)  # Ensure y0 is reshaped to a column vector
            y1 = np.dot(self.theta1, y0) + self.input_layer_bias  # Corrected bias shape
            y1 = self._sigmoid(y1)
            y2 = np.dot(self.theta2, y1) + self.hidden_layer_bias  # Corrected bias shape
            y2 = self._sigmoid(y2)
            
            # Backpropagation
            actual_vals = np.zeros((self.output_size, 1))
            actual_vals[data['label']] = 1
            output_errors = actual_vals - y2
            hidden_errors = np.dot(self.theta2.T, output_errors) * self._sigmoid_prime(y1)
            
            # Update weights and biases
            self.theta2 += self.LEARNING_RATE * np.dot(output_errors, y1.T)
            self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors, y0.T)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(self.theta1, np.mat(test).T) + self.input_layer_bias
        y1 = self._sigmoid(y1)
        y2 = np.dot(self.theta2, y1) + self.hidden_layer_bias
        y2 = self._sigmoid(y2)
        return np.argmax(y2)
    
    def save(self):
        if not self._use_file:
            return
        nn_data = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "input_layer_bias": self.input_layer_bias.tolist(),
            "hidden_layer_bias": self.hidden_layer_bias.tolist()
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile:
            json.dump(nn_data, nnFile)
    
    def _load(self):
        if not self._use_file or not os.path.isfile(OCRNeuralNetwork.NN_FILE_PATH):
            print(f"No existing neural network file found at {OCRNeuralNetwork.NN_FILE_PATH}. Initializing new network.")
            return
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'r') as nnFile:
            nn_data = json.load(nnFile)
        self.theta1 = np.array(nn_data['theta1'])
        self.theta2 = np.array(nn_data['theta2'])
        self.input_layer_bias = np.array(nn_data['input_layer_bias']).reshape(self.num_hidden_nodes, 1)
        self.hidden_layer_bias = np.array(nn_data['hidden_layer_bias']).reshape(self.output_size, 1)
