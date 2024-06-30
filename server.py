from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from ocr import OCRNeuralNetwork

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
CORS(app)

nn = OCRNeuralNetwork()

def preprocess_image(image_data):
    image_data = image_data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32')
    image_array = 255 - image_array  # Invert image
    image_array /= 255.0  # Normalize
    return image_array.flatten()

@app.route('/')
def index():
    return render_template('ocr.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    image_data = data['image']
    label = data['label']
    processed_image = preprocess_image(image_data)
    training_data = [{'y0': processed_image, 'label': label}]
    nn.train(training_data)
    nn.save()
    return jsonify({'message': 'Training successful'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    processed_image = preprocess_image(image_data)
    prediction = nn.predict(processed_image)
    return jsonify({'digit': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
