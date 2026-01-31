"""
Real-Time Right Leg Detection Application
Uses MediaPipe Pose + Segmentation for accurate leg detection
"""

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import os
from utils.leg_detector import RightLegDetector
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'png', 'jpg', 'jpeg'}

# Global variables
camera = None
video_path = None
processing_mode = 'webcam'
current_detector = None
detector_lock = False  # Prevent simultaneous detector access


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    global video_path, processing_mode
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        video_path = filepath
        processing_mode = 'video' if filename.lower().endswith(('.mp4', '.avi', '.mov')) else 'image'
        
        return jsonify({
            'success': True,
            'filename': filename,
            'mode': processing_mode
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    """Generate frames for video streaming"""
    global camera, video_path, processing_mode, current_detector, detector_lock
    
    # Determine if we need a new detector
    need_new_detector = (current_detector is None) or (processing_mode == 'image')
    
    if need_new_detector:
        # Close old detector if switching modes
        if current_detector is not None and processing_mode == 'image':
            try:
                current_detector.pose.close()
                current_detector.selfie_seg.close()
            except:
                pass
            current_detector = None
        
        # Create appropriate detector
        static_mode = (processing_mode == 'image')
        detector = RightLegDetector(static_mode=static_mode)
        
        # Only save to current_detector if NOT image mode (images get fresh detector each time)
        if processing_mode != 'image':
            current_detector = detector
    else:
        detector = current_detector
    
    try:
        if processing_mode == 'webcam':
            if camera is None:
                camera = cv2.VideoCapture(0)
                # Set camera properties for better performance
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap = camera
            
            while True:
                success, frame = cap.read()
                
                if not success:
                    print("Failed to read from webcam")
                    break
                
                # Process frame with right leg detection
                processed_frame = detector.process_frame(frame)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        elif processing_mode == 'video':
            cap = cv2.VideoCapture(video_path)
            
            while True:
                success, frame = cap.read()
                
                if not success:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Process frame with right leg detection
                processed_frame = detector.process_frame(frame)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        elif processing_mode == 'image':
            # For image, read once and loop
            frame = cv2.imread(video_path)
            if frame is not None:
                print(f"Processing image: {video_path}, shape: {frame.shape}")
                processed_frame = detector.process_frame(frame, debug=True)
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_bytes = buffer.tobytes()
                
                # Clean up image detector after processing
                try:
                    detector.pose.close()
                    detector.selfie_seg.close()
                except:
                    pass
                
                # Loop the same processed image
                while True:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                print(f"Failed to read image: {video_path}")
                
    except Exception as e:
        print(f"Error in generate_frames: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Only cleanup video capture if not webcam (webcam stays persistent)
        if processing_mode == 'video':
            try:
                cap.release()
            except:
                pass


@app.route('/switch_mode/<mode>')
def switch_mode(mode):
    """Switch between webcam and uploaded file"""
    global processing_mode, camera, current_detector
    
    if mode == 'webcam':
        processing_mode = 'webcam'
        
        # Release camera if exists
        if camera is not None:
            camera.release()
            camera = None
        
        # Clean up detector to force recreation
        if current_detector is not None:
            try:
                current_detector.pose.close()
                current_detector.selfie_seg.close()
            except:
                pass
            current_detector = None
            
        return jsonify({'success': True, 'mode': 'webcam'})
    
    return jsonify({'error': 'Invalid mode'}), 400


@app.route('/stop')
def stop():
    """Stop camera and release resources"""
    global camera, current_detector
    
    if camera is not None:
        camera.release()
        camera = None
    
    # Also cleanup detector
    if current_detector is not None:
        try:
            current_detector.pose.close()
            current_detector.selfie_seg.close()
        except:
            pass
        current_detector = None
        
    return jsonify({'success': True})


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)