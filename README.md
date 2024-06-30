# OCR Neural Network Project

Welcome to the OCR Neural Network Project! This project demonstrates a simple yet effective Optical Character Recognition (OCR) system using a neural network implemented from scratch. The project includes components for training a neural network on the MNIST dataset and a web-based interface that allows users to draw digits and get real-time predictions.

## Components

1. **Client (ocr.js)**: Handles the front-end logic, including the drawing interface on the canvas, sending data to the server, and displaying results.
2. **Server (server.py)**: Manages HTTP requests, processes images, and communicates with the neural network for training and prediction.
3. **User Interface (ocr.html)**: Provides an intuitive interface for users to draw digits, train the neural network, and get predictions.
4. **Neural Network (ocr.py)**: Implements a neural network with backpropagation for OCR tasks.
5. **Neural Network Design Script (neural_network_design.py)**: Script for designing and initializing the neural network (not actively used in this demonstration).

## Features

- **Drawing Canvas**: Users can draw digits on a 280x280 pixel canvas which gets resized to 28x28 for neural network processing.
- **Training**: Train the neural network with user-provided digits and labels through the web interface.
- **Prediction**: Real-time digit recognition and prediction.
- **Persistence**: Saves and loads neural network parameters from a JSON file (`nn.json`).

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/yourusername/ocr-neural-network.git
cd ocr-neural-network

Install Dependencies
Ensure you have Python installed, then install the required packages:

pip install numpy pillow tqdm tensorflow flask

Run the Server
Start the server to handle HTTP requests:

python server.py

Open the Web Interface
Open a web browser and navigate to http://127.0.0.1:5000 to access the drawing canvas and interact with the neural network.

