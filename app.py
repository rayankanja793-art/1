from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
# Load the model once when the server starts
model = YOLO('yolo11n.pt') 

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get the image from the request
    file = request.files['image'].read()
    np_img = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # 2. Perform Inference
    results = model(frame)
    
    # 3. Process detections (Extract game data)
    # Example: Count how many 'players' were detected
    player_count = sum(1 for box in results[0].boxes if box.cls == 0) # Assuming class 0 is player
    
    # 4. Return as JSON
    return jsonify({"player_count": player_count, "prediction": "active_play"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
