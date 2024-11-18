from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the pre-trained model from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"
model = hub.load(MODEL_URL)

# Load ImageNet class labels
with open('imagenet_classes.txt') as f:
    CLASS_NAMES = f.read().splitlines()

def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using the model
    predictions = model(img_array).numpy()[0]  # Convert predictions to NumPy array
    class_idx = np.argmax(predictions)  # Find the index of the highest probability
    confidence = predictions[class_idx]  # Get the highest confidence

    # Retrieve the class name
    class_name = CLASS_NAMES[class_idx-1]

    return class_name, confidence


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        # Ensure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Make a prediction
        class_name, confidence = predict_image(file_path)
        return render_template('result.html', class_name=class_name, confidence=confidence, image_path=file_path)

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)