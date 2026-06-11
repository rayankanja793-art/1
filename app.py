from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load model - it will auto-download if not present
try:
    model = YOLO('yolo11n.pt')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check if the 'image' key exists in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 2. Convert file to CV2 format
    try:
        np_img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        # 3. Perform Inference
        results = model(frame)
        
        # 4. Count players (class 0 is usually 'person' in YOLO)
        player_count = sum(1 for box in results[0].boxes if box.cls == 0)
        
        return jsonify({"player_count": player_count, "status": "success"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
