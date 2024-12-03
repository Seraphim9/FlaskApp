from flask import Flask, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
import io
import numpy as np
import torch
from PIL import Image
import cv2
import os
import logging
import tempfile

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Inisialisasi logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Buat folder jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class Detection:
    def __init__(self):
        try:
            print("Initializing YOLOv5 model...")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='s_student.pt')
            self.model.eval()
            print("Model initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def predict(self, img, classes=None, conf=0.5):
        if not hasattr(self, 'model'):
            raise AttributeError("Model not properly initialized")
        results = self.model(img)
        return results

    def predict_and_detect(self, img, classes=None, conf=0.5, rectangle_thickness=2, text_thickness=2):
        results = self.predict(img, classes, conf=conf)
        result_pandas = results.pandas().xyxy[0]
        
        for idx, detection in result_pandas.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            conf_score = detection['confidence']
            class_name = detection['name']
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), rectangle_thickness)
            label = f"{class_name} {conf_score:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
        
        return img, results

    def detect_from_image(self, image):
        result_img, results = self.predict_and_detect(image, conf=0.5)
        return result_img


# Initialize detection object
detection = Detection()

@app.route('/')
def index():
    return render_template('index.html')

# Image detection endpoint
@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    if file:
        try:
            # Check if the file is an image
            filename = secure_filename(file.filename)
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                return 'Invalid image file format'

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img = Image.open(file_path).convert("RGB")
            img = np.array(img)
            img = cv2.resize(img, (512, 512))
            img = detection.detect_from_image(img)
            output = Image.fromarray(img)

            buf = io.BytesIO()
            output.save(buf, format="PNG")
            buf.seek(0)

            os.remove(file_path)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            print(f"Error in apply_detection: {str(e)}")
            return f"Error processing image: {str(e)}"

# Video detection endpoint
@app.route('/video-detection/', methods=['POST'])
def apply_video_detection():
    if 'video' not in request.files:
        return 'No file part'

    file = request.files['video']
    if file.filename == '':
        return 'No selected file'

    if file:
        try:
            # Check if the file is a video
            filename = secure_filename(file.filename)
            if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
                return 'Invalid video file format'

            # Save uploaded video to temporary file
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, filename)
            output_path = os.path.join(temp_dir, 'output_' + filename)
            file.save(input_path)

            # Open video file
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return 'Error opening video file'

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply detection
                frame = detection.detect_from_image(frame)
                out.write(frame)

            # Release resources
            cap.release()
            out.release()

            # Send processed video file
            return send_file(output_path, 
                           mimetype='video/mp4',
                           as_attachment=False,
                           download_name='processed_' + filename)

        except Exception as e:
            print(f"Error in apply_video_detection: {str(e)}")
            return f"Error processing video: {str(e)}"
        finally:
            # Cleanup temporary files
            try:
                os.remove(input_path)
                os.remove(output_path)
                os.rmdir(temp_dir)
            except:
                pass

# Webcam routes
@app.route('/video')
def index_video():
    return render_template('video.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frame = cv2.resize(frame, (512, 512))
            frame = detection.detect_from_image(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in gen_frames: {str(e)}")
            break
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)