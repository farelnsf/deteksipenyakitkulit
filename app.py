from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import base64
import io
import os
from ultralytics import YOLO
from datetime import datetime
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB file size limit
model = YOLO('weights/best.pt')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

def convert_to_percentage(box, img_width, img_height):
    x1, y1, x2, y2 = box
    return [
        round((x1 / img_width) * 100, 2),
        round((y1 / img_height) * 100, 2),
        round((x2 / img_width) * 100, 2),
        round((y2 / img_height) * 100, 2)
    ]

@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera_page():
    return render_template('capture.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect():
    detection_results = {
        'result': None,
        'labels': [],
        'scores': [],
        'boxes': [],
        'pixel_boxes': [],
        'has_detection': False,
        'image_size': None
    }

    try:
        if 'image' in request.form:
            image_data = request.form['image'].split(',')[1]
            img_bytes = base64.b64decode(image_data)
            npimg = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('error.html', error_message="No file selected"), 400
            if not allowed_file(file.filename):
                return render_template('error.html', error_message="Invalid file type. Only JPG, JPEG, and PNG are allowed."), 400
            npimg = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if frame is None:
                return render_template('error.html', error_message="Invalid image format"), 400
        else:
            return render_template('error.html', error_message="No image provided"), 400

        if frame.shape[0] < 300 or frame.shape[1] < 300:
            return render_template('error.html', error_message="Image resolution too low (min 300x300px)"), 400

        img_height, img_width = frame.shape[:2]
        detection_results['image_size'] = {'width': img_width, 'height': img_height}

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=frame_rgb, save=False, conf=0.25, imgsz=640, augment=True)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            names = model.names
            for box, cls_id, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                                         results[0].boxes.cls.cpu().numpy(),
                                         results[0].boxes.conf.cpu().numpy()):
                cls_id = int(cls_id)
                conf = float(conf)
                box_pixels = [int(round(x)) for x in box.tolist()]
                detection_results['labels'].append(names[cls_id])
                detection_results['scores'].append(f"{conf * 100:.1f}%")
                detection_results['pixel_boxes'].append(box_pixels)
                detection_results['boxes'].append(convert_to_percentage(box, img_width, img_height))
            detection_results['has_detection'] = True

        # Annotate result
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # Encode result in memory (base64)
        _, buffer = cv2.imencode('.jpg', annotated)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        detection_results['result'] = f"data:image/jpeg;base64,{encoded_image}"

        return render_template('result.html', **detection_results)

    except Exception as e:
        app.logger.error(f"Error during detection: {str(e)}")
        return render_template('error.html', error_message=f"Error processing image: {str(e)}")

@app.route('/realtime_view')
def realtime_view():
    return render_template('realtime.html')

def generate_frames(device_id='0'):
    cap = None
    try:
        # Try to open camera with proper device ID
        try:
            device_id_int = int(device_id)
            cap = cv2.VideoCapture(device_id_int)
        except ValueError:
            cap = cv2.VideoCapture(device_id)

        if not cap.isOpened():
            app.logger.info(f"Device ID from request: {device_id}")
            raise RuntimeError("Could not open camera")
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while True:
            success, frame = cap.read()
            if not success:
                app.logger.error("Failed to read frame from camera")
                break

            # Process frame with YOLO model
            height, width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use stream=True for continuous processing
            results = model.predict(source=frame_rgb, stream=True, conf=0.25, imgsz=640, augment=True)
            
            annotated = frame.copy()
            for result in results:
                if result.boxes is not None:
                    for box, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                        x1, y1, x2, y2 = box.tolist()
                        box_percent = convert_to_percentage([x1, y1, x2, y2], width, height)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Create informative label
                        label = (f"{model.names[int(cls_id)]} {conf*100:.1f}% | "
                                f"X:{box_percent[0]:.1f}%, Y:{box_percent[1]:.1f}% | "
                                f"W:{box_percent[2]-box_percent[0]:.1f}%, H:{box_percent[3]-box_percent[1]:.1f}%")
                        
                        # Put text on image
                        cv2.putText(annotated, label, (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                app.logger.error("Failed to encode frame")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except Exception as e:
        app.logger.error(f"Error in video feed: {str(e)}")
        
        # Create error frame
        error_frame = np.zeros((300, 500, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_frame, str(e), (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    finally:
        if cap and cap.isOpened():
            cap.release()

@app.route('/realtime_feed')
def realtime_feed():
    device = request.args.get('device', '0')
    return Response(generate_frames(device), 
                  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_message="Internal server error"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)