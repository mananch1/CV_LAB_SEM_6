import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------
# Load Face Detector (Caffe)
# ---------------------------
net = cv2.dnn.readNetFromCaffe(
    "..\\deploy.prototxt",
    "..\\res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# ---------------------------
# New MediaPipe Tasks API Setup
# ---------------------------
# Download 'face_landmarker.task' from the MediaPipe website
base_options = python.BaseOptions(model_asset_path='..\\face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

prev_gray = None

# ---------------------------
# Decode base64 frame
# ---------------------------
def decode_base64_frame(data):
    header, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

# ---------------------------
# Face detection (Caffe)
# ---------------------------
def detect_largest_face(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123)
    )
    net.setInput(blob)
    detections = net.forward()

    largest_face = None
    largest_area = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            pad = int(0.15 * (x2 - x1))
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_face = (x1, y1, x2, y2)

    return largest_face

# ---------------------------
# Crop face
# ---------------------------
def crop_face(frame, box):
    x1, y1, x2, y2 = box
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return face

# ---------------------------
# Process Frame Landmarks (New Addition)
# ---------------------------
def get_face_landmarks(frame):
    """Processes the frame once and returns the landmarks to be used by other functions."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Tasks API requires an 'mp.Image' object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    results = face_landmarker.detect(mp_image)
    
    if not results.face_landmarks:
        return None
        
    return results.face_landmarks[0] # Return the first face's landmarks

# ---------------------------
# Eye Aspect Ratio
# ---------------------------
def eye_aspect_ratio(eye):
    p1, p2, p3, p4, p5, p6 = eye
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2 * h)

# ---------------------------
# Blink detection
# ---------------------------
def blink_detection(landmarks, frame_shape):
    h, w, _ = frame_shape
    left_eye = []
    right_eye = []

    for idx in LEFT_EYE:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        left_eye.append(np.array([x, y]))

    for idx in RIGHT_EYE:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        right_eye.append(np.array([x, y]))

    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)
    
    EAR = (ear_left + ear_right) / 2
    return EAR

# ---------------------------
# Head pose (yaw)
# ---------------------------
def head_pose_estimation(landmarks, frame_shape):
    h, w, _ = frame_shape

    nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])

    yaw = np.linalg.norm(left_eye - nose) - np.linalg.norm(right_eye - nose)
    return yaw

# ---------------------------
# Optical flow motion
# ---------------------------
def optical_flow_motion(frame):
    global prev_gray

    # 1. FORCE a consistent size so Farneback never gets mismatched frames
    standardized_frame = cv2.resize(frame, (150, 150))
    
    gray = cv2.cvtColor(standardized_frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        return 0

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))

    prev_gray = gray

    return magnitude