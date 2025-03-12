import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# No longer using HTTPBasicAuth, handling auth in the route
users = {
    "admin": generate_password_hash("password123")
}

MODEL_PATH = os.environ.get('MODEL_PATH', 'cifar10_mobilenet_final.h5')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("model loaded bro")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image):
    image = image.resize((32, 32))
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": "CIFAR-10 classifier"})

@app.route('/predict', methods=['POST', 'GET']) #added get method
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    username = request.form.get('username')
    password = request.form.get('password')

    if username not in users or not check_password_hash(users[username], password):
        return jsonify({"error": "Invalid credentials"}), 401

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return jsonify({"error": "Invalid file type. Only image files are allowed"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        return jsonify({
            "class": predicted_class,
            "class_index": int(predicted_class_index),
            "confidence": confidence,
            "all_probabilities": {
                class_names[i]: float(predictions[0][i])
                for i in range(len(class_names))
            }
        })

    except IOError:
        return jsonify({"error": "Unable to open image file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def info():
    return jsonify({
        "name": "CIFAR-10 Image Classification API",
        "description": "API for classifying images into one of ten CIFAR-10 classes",
        "endpoints": {
            "/predict": "POST - Submit an image for classification (requires authentication)",
            "/health": "GET - Check API health status"
        },
        "classes": class_names
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) #debug = true is added