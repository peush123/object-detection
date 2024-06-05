from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
import numpy as np
import os
import imutils
import time
import cv2
from imutils.video import VideoStream
from subprocess import Popen

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECT_FOLDER'] = 'runs/detect'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECT_FOLDER'], exist_ok=True)

# Define the class labels for MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Set paths for MobileNet SSD model files
PROTOTXT_PATH = r"C:\Users\pkspi\OneDrive\Desktop\Object-Detection-Web-App-Using-YOLOv7-and-Flask-main\MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = r"C:\Users\pkspi\OneDrive\Desktop\Object-Detection-Web-App-Using-YOLOv7-and-Flask-main\MobileNetSSD_deploy.caffemodel"
CONFIDENCE_THRESHOLD = 0.2

# Load MobileNet SSD model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Get image data from the frontend
    image_data = request.files['image']
    
    # Convert the image data to a format suitable for OpenCV
    image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    
    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    
    results = []
    
    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            results.append({"label": CLASSES[idx], "confidence": float(confidence), "box": [int(startX), int(startY), int(endX), int(endY)]})
            
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    ret, jpeg = cv2.imencode('.jpg', image)
    response = jsonify(results)
    response.status_code = 200
    
    return response

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)
            
            # Execute the YOLOv7 detection process
            process = Popen(["python", "yolov7/detect.py", '--source', filepath, "--weights", "yolov7/yolov7.pt"], shell=True)
            process.wait()
            
            # Get the path to the detected image
            folder_path = app.config['DETECT_FOLDER']
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
            image_path = os.path.join(folder_path, latest_subfolder, filename)
            
            return render_template('index.html', image_path=image_path)
    
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    # Initialize the video stream with DirectShow backend
    vs = VideoStream(src=0, backend='ds').start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # Prepare the frame for detection
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections and draw bounding boxes
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_THRESHOLD:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
