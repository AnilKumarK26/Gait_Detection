# from flask import Flask, render_template, Response, request, jsonify
# import cv2
# import numpy as np
# import os
# import time
# import threading
# from utils.leg_detector import RightLegDetector
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER']      = 'static/uploads'
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024          # 100 MB
# app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'png', 'jpg', 'jpeg'}

# # ─── Thread-safe shared state ─────────────────────────────────────
# _lock = threading.Lock()

# class _AppState:
#     """All mutable shared state lives here, protected by _lock."""
#     processing_mode: str            = 'webcam'   # 'webcam' | 'video' | 'image'
#     video_path:      str | None     = None
#     stop_requested:  bool           = False      # set by /stop, cleared by generator
#     generation:      int            = 0          # bumped on every new /video_feed

# _state = _AppState()


# def _allowed_file(filename: str) -> bool:
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# # ─── ROUTES ───────────────────────────────────────────────────────

# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     if not _allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file type'}), 400

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)

#     is_video = filename.lower().endswith(('.mp4', '.avi', '.mov'))

#     with _lock:
#         _state.video_path      = filepath
#         _state.processing_mode = 'video' if is_video else 'image'
#         _state.stop_requested  = False
#         _state.generation     += 1          # invalidate any running generator

#     return jsonify({
#         'success':  True,
#         'filename': filename,
#         'mode':     _state.processing_mode,
#     })


# @app.route('/video_feed')
# def video_feed():
#     # Bump generation so any previous generator will exit on next iteration
#     with _lock:
#         _state.stop_requested = False
#         _state.generation    += 1
#         my_generation = _state.generation

#     return Response(
#         _generate_frames(my_generation),
#         mimetype='multipart/x-mixed-replace; boundary=frame',
#     )


# @app.route('/switch_mode/webcam')
# def switch_mode_webcam():
#     with _lock:
#         _state.processing_mode = 'webcam'
#         _state.stop_requested  = False
#         _state.generation     += 1
#     return jsonify({'success': True, 'mode': 'webcam'})


# @app.route('/stop')
# def stop():
#     with _lock:
#         _state.stop_requested = True
#     return jsonify({'success': True})


# @app.route('/gait_data')
# def gait_data_endpoint():
#     """Return the most recent gait data (polled by the frontend)."""
#     with _lock:
#         data = getattr(_state, 'last_gait_data', None)
#     return jsonify(data or {"status": "no data yet"})


# # ─── FRAME GENERATOR ──────────────────────────────────────────────

# _FPS_LIMIT = 30
# _FRAME_INTERVAL = 1.0 / _FPS_LIMIT


# def _generate_frames(my_generation: int):
#     """
#     Generator that owns its own detector and VideoCapture.
#     Exits when:
#       • the client disconnects (GeneratorExit)
#       • /stop was called (_state.stop_requested)
#       • a newer /video_feed request supersedes us (_state.generation != my_generation)
#     """
#     cap     = None
#     detector = None

#     try:
#         # ── snapshot current mode ──────────────────────────────────
#         with _lock:
#             mode       = _state.processing_mode
#             video_path = _state.video_path

#         # ── create detector (owned by this generator) ─────────────
#         static = (mode == 'image')
#         detector = RightLegDetector(static_mode=static)

#         # ── open capture if needed ─────────────────────────────────
#         if mode == 'webcam':
#             cap = cv2.VideoCapture(0)
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#             cap.set(cv2.CAP_PROP_FPS, _FPS_LIMIT)

#         elif mode == 'video':
#             if video_path is None:
#                 return
#             cap = cv2.VideoCapture(video_path)

#         # ── IMAGE mode: render once, loop the static result ────────
#         if mode == 'image':
#             if video_path is None:
#                 return
#             frame = cv2.imread(video_path)
#             if frame is None:
#                 return

#             rendered, gait = detector.process_frame(frame, debug=True)
#             with _lock:
#                 _state.last_gait_data = gait

#             _, buf = cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY, 90])
#             payload = buf.tobytes()

#             while True:
#                 # Check exit conditions
#                 with _lock:
#                     if _state.stop_requested or _state.generation != my_generation:
#                         return
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')
#                 time.sleep(_FRAME_INTERVAL)

#         # ── WEBCAM / VIDEO loop ────────────────────────────────────
#         while True:
#             # Exit conditions
#             with _lock:
#                 if _state.stop_requested or _state.generation != my_generation:
#                     return

#             ok, frame = cap.read()
#             if not ok:
#                 if mode == 'video':
#                     # loop video
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                     continue
#                 else:
#                     # webcam broken
#                     return

#             rendered, gait = detector.process_frame(frame, debug=True)
#             with _lock:
#                 _state.last_gait_data = gait

#             _, buf = cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

#             time.sleep(_FRAME_INTERVAL)

#     except GeneratorExit:
#         # Client disconnected — clean exit
#         pass
#     except Exception as exc:
#         import traceback
#         traceback.print_exc()
#     finally:
#         # ── guaranteed cleanup ─────────────────────────────────────
#         if cap is not None:
#             cap.release()
#         if detector is not None:
#             detector.close()


# # ─── STARTUP ──────────────────────────────────────────────────────

# if __name__ == '__main__':
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

"""
Real-Time Right Leg Detection — Flask backend (Multi-Person Support)
Uses connected components to detect multiple people.
"""

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import os
import time
import threading
from utils.multi_person_leg_detector import MultiPersonRightLegDetector
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER']      = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024          # 100 MB
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'png', 'jpg', 'jpeg'}

# ─── Thread-safe shared state ─────────────────────────────────────
_lock = threading.Lock()

class _AppState:
    """All mutable shared state lives here, protected by _lock."""
    processing_mode: str            = 'webcam'   # 'webcam' | 'video' | 'image'
    video_path:      str | None     = None
    stop_requested:  bool           = False      # set by /stop, cleared by generator
    generation:      int            = 0          # bumped on every new /video_feed

_state = _AppState()


def _allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ─── ROUTES ───────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not _allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    is_video = filename.lower().endswith(('.mp4', '.avi', '.mov'))

    with _lock:
        _state.video_path      = filepath
        _state.processing_mode = 'video' if is_video else 'image'
        _state.stop_requested  = False
        _state.generation     += 1          # invalidate any running generator

    return jsonify({
        'success':  True,
        'filename': filename,
        'mode':     _state.processing_mode,
    })


@app.route('/video_feed')
def video_feed():
    # Bump generation so any previous generator will exit on next iteration
    with _lock:
        _state.stop_requested = False
        _state.generation    += 1
        my_generation = _state.generation

    return Response(
        _generate_frames(my_generation),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/switch_mode/webcam')
def switch_mode_webcam():
    with _lock:
        _state.processing_mode = 'webcam'
        _state.stop_requested  = False
        _state.generation     += 1
    return jsonify({'success': True, 'mode': 'webcam'})


@app.route('/stop')
def stop():
    with _lock:
        _state.stop_requested = True
    return jsonify({'success': True})


@app.route('/gait_data')
def gait_data_endpoint():
    """Return the most recent gait data (polled by the frontend)."""
    with _lock:
        data = getattr(_state, 'last_gait_data', None)
    return jsonify(data or {"status": "no data yet"})


# ─── FRAME GENERATOR ──────────────────────────────────────────────

_FPS_LIMIT = 30
_FRAME_INTERVAL = 1.0 / _FPS_LIMIT


def _generate_frames(my_generation: int):
    """
    Generator that owns its own detector and VideoCapture.
    Now supports multi-person detection.
    """
    cap     = None
    detector = None

    try:
        # ── snapshot current mode ──────────────────────────────────
        with _lock:
            mode       = _state.processing_mode
            video_path = _state.video_path

        # ── create multi-person detector ───────────────────────────
        static = (mode == 'image')
        detector = MultiPersonRightLegDetector(static_mode=static)

        # ── open capture if needed ─────────────────────────────────
        if mode == 'webcam':
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, _FPS_LIMIT)

        elif mode == 'video':
            if video_path is None:
                return
            cap = cv2.VideoCapture(video_path)

        # ── IMAGE mode: render once, loop the static result ────────
        if mode == 'image':
            if video_path is None:
                return
            frame = cv2.imread(video_path)
            if frame is None:
                return

            rendered, gait_list = detector.process_frame(frame, debug=True)
            with _lock:
                _state.last_gait_data = {
                    "num_people": len(gait_list),
                    "people": gait_list
                }

            _, buf = cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY, 90])
            payload = buf.tobytes()

            while True:
                # Check exit conditions
                with _lock:
                    if _state.stop_requested or _state.generation != my_generation:
                        return
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')
                time.sleep(_FRAME_INTERVAL)

        # ── WEBCAM / VIDEO loop ────────────────────────────────────
        while True:
            # Exit conditions
            with _lock:
                if _state.stop_requested or _state.generation != my_generation:
                    return

            ok, frame = cap.read()
            if not ok:
                if mode == 'video':
                    # loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # webcam broken
                    return

            rendered, gait_list = detector.process_frame(frame, debug=True)
            with _lock:
                _state.last_gait_data = {
                    "num_people": len(gait_list),
                    "people": gait_list
                }

            _, buf = cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

            time.sleep(_FRAME_INTERVAL)

    except GeneratorExit:
        # Client disconnected — clean exit
        pass
    except Exception as exc:
        import traceback
        traceback.print_exc()
    finally:
        # ── guaranteed cleanup ─────────────────────────────────────
        if cap is not None:
            cap.release()
        if detector is not None:
            detector.close()


# ─── STARTUP ──────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)