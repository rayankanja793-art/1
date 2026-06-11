from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
# Load the model once
model = YOLO('yolo11n.pt') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image'].read()
    np_img = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Run Inference
    results = model(frame)
    # Simple logic: detect players (class 0)
    count = sum(1 for box in results[0].boxes if box.cls == 0)
    
    return jsonify({"player_count": count})

if __name__ == '__main__':
    app.run()
