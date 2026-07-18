import os
import cv2
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load Haar Cascade
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

# Fallback just in case OpenCV data path fails
if face_cascade.empty():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(local_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(filepath, filename):
    image = cv2.imread(filepath)
    if image is None: return None, 0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
    
    # Draw neon boxes
    for (x, y, w, h) in faces:
        color = (255, 105, 180) # Hot Pink in BGR for a cyberpunk feel
        thickness = 3
        # Primary box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
        
        # Corner accents
        length = 25
        cv2.line(image, (x, y), (x+length, y), color, thickness+2)
        cv2.line(image, (x, y), (x, y+length), color, thickness+2)
        cv2.line(image, (x+w, y), (x+w-length, y), color, thickness+2)
        cv2.line(image, (x+w, y), (x+w, y+length), color, thickness+2)
        cv2.line(image, (x, y+h), (x+length, y+h), color, thickness+2)
        cv2.line(image, (x, y+h), (x, y+h-length), color, thickness+2)
        cv2.line(image, (x+w, y+h), (x+w-length, y+h), color, thickness+2)
        cv2.line(image, (x+w, y+h), (x+w, y+h-length), color, thickness+2)
        
        cv2.putText(image, "TARGET", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], "processed_" + filename)
    cv2.imwrite(output_path, image)
    return output_path, len(faces)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_faces():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        processed_path, face_count = process_image(filepath, filename)
        
        if processed_path:
            return jsonify({
                'success': True,
                'image_url': '/' + processed_path,
                'face_count': face_count
            })
        else:
            return jsonify({'error': 'Failed to process image'}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5002) # Running on 5002 to avoid conflicts
