from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, send_from_directory
import joblib
import numpy as np
import google.generativeai as genai
import os
from werkzeug.utils import secure_filename
import torch
from dotenv import load_dotenv
from ultralytics import YOLO
import threading
import cv2
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import torch
from pathlib import Path
import time
import json
import torch.nn as nn
from io import BytesIO
import torchvision.models as models
import uuid
import timm

app = Flask(__name__)

# Load the models
model1 = joblib.load("Classification1.joblib")
model2 = joblib.load("Regression.joblib")



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash') 

@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/Predict.html')
def predict_page():
    return render_template('Predict.html')
@app.route('/Ai.html')
def chatbot():
    return render_template('Ai.html')
@app.route('/live.html')
def video_detection():
    return render_template('live.html')
@app.route('/camera.html')
def camera_detection():
    return render_template('camera.html')
@app.route('/Satellite.html')
def satellite():
    return render_template('Satellite.html')

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the YOLOv11 model
model_path = "C:/Users/anugr/Downloads/Forest_fire/wildfire-detection/runs/detect/train/weights/best.pt"
model3 = YOLO(model_path)

# Processing variables
processing = False
video_path = ''
processed_video_path = ''

# Camera variables
camera_active = False
cap = None
frame_skip = 2  # Process every nth frame to reduce lag
frame_counter = 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video():
    global processing, video_path, processed_video_path
    
    # Get original filename and create output filename
    original_filename = os.path.basename(video_path)
    output_filename = os.path.splitext(original_filename)[0] + '.avi'
    processed_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'predict', output_filename)
    
    # Create predict directory if it doesn't exist
    os.makedirs(os.path.join(app.config['PROCESSED_FOLDER'], 'predict'), exist_ok=True)
    
    # Process video with YOLO's optimized method
    model3.predict(video_path, save=True, project=PROCESSED_FOLDER, name='predict', exist_ok=True)
    
    processing = False

def camera_thread():
    global camera_active, cap, frame_counter
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue
            
        # Make detections
        results = model3(frame, verbose=False)
        
        # Process results
        for result in results:
            frame = result.plot()
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    cap = None

@app.route('/video_feed')
def video_feed():
    return Response(camera_thread(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control_camera', methods=['POST'])
def control_camera():
    global camera_active, cap
    
    action = request.form.get('action')
    
    if action == 'start' and not camera_active:
        camera_active = True
        return jsonify({'status': 'success', 'message': 'Camera started'})
    elif action == 'stop' and camera_active:
        camera_active = False
        if cap is not None:
            cap.release()
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid action'})

@app.route('/upload_and_process', methods=['POST'])
def upload_and_process():
    global processing, video_path
    
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type'})
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    
    # Start processing
    processing = True
    threading.Thread(target=process_video).start()
    
    return jsonify({'status': 'success', 'message': 'Processing started'})

@app.route('/get_status')
def get_status():
    global processing, processed_video_path
    
    # Check if processing is complete and file exists
    video_ready = False
    download_path = ''
    
    if not processing:
        # Check for the processed file with .avi extension
        original_filename = os.path.basename(video_path)
        output_filename = os.path.splitext(original_filename)[0] + '.avi'
        predicted_path = os.path.join(app.config['PROCESSED_FOLDER'], 'predict', output_filename)
        
        if os.path.exists(predicted_path):
            video_ready = True
            processed_video_path = predicted_path
            download_path = f'/download/predict/{output_filename}'
    
    return jsonify({
        'processing': processing,
        'video_ready': video_ready,
        'download_path': download_path
    })

@app.route('/download/predict/<filename>')
def download_file(filename):
    return send_from_directory(
        os.path.join(app.config['PROCESSED_FOLDER'], 'predict'),
        filename,
        as_attachment=True
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        rain = float(request.form.get('rain', 0.0))
        ffmc = float(request.form.get('ffmc', 0.0))
        dmc = float(request.form.get('dmc', 0.0))
        isi = float(request.form.get('isi', 0.0))

        # Prepare input data
        input_data = np.array([[rain, ffmc, dmc, isi]])

        # Make predictions
        prediction = model1.predict(input_data)[0]
        prediction2 = model2.predict(input_data)[0]
        value=prediction2*3.5

        # Scale chance of fire (adjust based on your model's output range)
        chance_of_fire = min(max(value , 0), 100)  # Example: scale 0-5 to 0-100

        # Prepare response
        result = {
            'fire_detected': bool(prediction == 1 and value>=50),
            'chance_of_fire': round(chance_of_fire, 2),
            'progress_color': 'green' if chance_of_fire < 50 else 'red'
        }
        return jsonify(result)
    return render_template('Predict.html')
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Please provide a message"}), 400
    
    message = data['message'].strip()

    # Query Gemini API for response
    try:
        response = model.generate_content(f"Answer this question about Wildfires: {message}")
        answer = response.text  # Adjust based on Gemini API response structure
        return jsonify({"text": answer})
    except Exception as e:
        return jsonify({"error": f"Failed to get response: {str(e)}"}), 500

#SATELLITE

# Load environment variables
load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")

# Define the model architecture matching your saved weights
class WildfireDetector(nn.Module):
    def __init__(self):
        super(WildfireDetector, self).__init__()
        # Load pretrained ResNet50 as base model
        self.base_model = models.resnet50(weights=None)
        
        # Get the number of features from the last layer
        num_ftrs = self.base_model.fc.in_features
        
        # Replace the final fully connected layer with your custom layers
        self.base_model.fc = nn.Identity()  # Remove original FC layer
        
        # Add custom layers matching your saved model's dimensions
        self.fc1 = nn.Linear(num_ftrs, 512)    # 2048 -> 512
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)         # 512 -> 256
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)         # 256 -> 128
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.base_model(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Satellite_model = WildfireDetector()  # Create model instance
state_dict = torch.load("wildfire_satellite_detection_model.pth")  # Load state dict
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('base_model.fc.')}  # Load weights into model
Satellite_model.load_state_dict(state_dict)
Satellite_model.eval()  # Set to evaluation mode
Satellite_model.to(device)  # Move to appropriate device

# Define image transformations
preprocess_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess_transforms(img)
    return img_tensor.unsqueeze(0)

def satellite_cnn_predict(latitude, longitude, output_size, zoom_level, crop_amount, save_path):
    output_size_modified = (output_size[0], output_size[1] + crop_amount)
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{longitude},{latitude},{zoom_level}/{output_size_modified[0]}x{output_size_modified[1]}?access_token={MAPBOX_TOKEN}"
    response = requests.get(url)

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        remove_pixels = crop_amount
        remove_pixels_half = remove_pixels // 2

        img_cropped = img.crop(
            (0, remove_pixels_half, img.width, img.height - remove_pixels_half)
        )
        img_cropped.save(save_path)
        print(f"Image saved as '{save_path}'")

        processed_image = preprocess_image(save_path)
        processed_image = processed_image.to(device)

        with torch.no_grad():
            prediction = Satellite_model(processed_image)
            probability = torch.sigmoid(prediction).item()

        return probability
    else:
        print("Failed to retrieve the image.")
        return None


@app.route("/satellite_predict", methods=["POST"])
def satellite_predict():
    data = request.json
    latitude = data["location"][1]
    longitude = data["location"][0]
    zoom = data["zoom"]

    output_size = (350, 350)
    crop_amount = 35
    filename= f"satellite_{uuid.uuid4().hex}.png"
    save_path = f"satellite_images/{filename}"

    prediction_sattelite = satellite_cnn_predict(
        latitude,
        longitude,
        output_size=output_size,
        zoom_level=zoom,
        crop_amount=crop_amount,
        save_path=save_path,
    )

    if prediction_sattelite is None:
        return jsonify({"error": "Failed to process satellite image"}), 500

    satellite_confidence = round(
        (prediction_sattelite if prediction_sattelite > 0.5 else 1 - prediction_sattelite) * 100
    )
    satellite_status = 1 if prediction_sattelite > 0.5 else 0

    response = {
        "satellite_probability": satellite_confidence,
        "satellite_status": satellite_status,
    }

    return jsonify(response), 200
# camera Detection

class SimpleFireClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleFireClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes))
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model4 = SimpleFireClassifier(num_classes=3)
model4.load_state_dict(torch.load('camera_detection.pt', map_location=device))
model4 = model4.to(device)
model4.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Correct class mapping based on your target_to_class
CLASS_MAPPING = {
    0: "Smoke",
    1: "Fire",
    2: "No_Fire"
}

@app.route('/image_detection', methods=['POST'])
def image_detection():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    try:
        # Preprocess the image
        image = Image.open(BytesIO(image_file.read())).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get probabilities and prediction
        with torch.no_grad():
            outputs = model4(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        predicted_label = CLASS_MAPPING.get(predicted_class, "Unknown")
        
        return jsonify({
            'status': 'success',
            'prediction': predicted_label,
            'confidence': round(probabilities[predicted_class].item() * 100),
            'probabilities': {
                'Smoke': round(probabilities[0].item() * 100, 1),
                'Fire': round(probabilities[1].item() * 100, 1),
                'No_Fire': round(probabilities[2].item() * 100, 1)
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)