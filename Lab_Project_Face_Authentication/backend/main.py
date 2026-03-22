import cv2
import uvicorn
import time
import sys
import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from facenet_pytorch import InceptionResnetV1

sys.path.append(os.path.abspath('..'))
from livenessnet import LivenessNet

from fastapi import FastAPI, WebSocket

from utils import (
    decode_base64_frame,
    detect_largest_face,
    crop_face,
    get_face_landmarks,
    blink_detection
)

app = FastAPI()

# ---------------------------
# Tracking variables
# ---------------------------

tracker = None
box = None
frame_count = 0

# ---------------------------
# Blink detection
# ---------------------------

blink_frames = 0
EAR_THRESHOLD = 0.22
BLINK_MIN_FRAMES = 4

# ---------------------------
# PyTorch Liveness Model
# ---------------------------
print("Loading PyTorch liveness model...")
le_path = os.path.join("..", "le.pickle")
model_path = os.path.join("..", "liveness.model")
le = pickle.loads(open(le_path, "rb").read())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
liveness_model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
liveness_model.load_state_dict(torch.load(model_path, map_location=device))
liveness_model.to(device)
liveness_model.eval()
nn_transform = transforms.Compose([transforms.ToTensor()])

# ---------------------------
# Face Authentication Model
# ---------------------------
print("Loading FaceNet Siamese Model...")
auth_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
registered_password = None
is_registered = False

registration_embeddings = []
REGISTRATION_FRAMES_NEEDED = 10
registered_anchor = None

# ---------------------------
# Liveness session
# ---------------------------

liveness_verified = False
liveness_timestamp = 0

LIVENESS_TIMEOUT = 30  # seconds

# ---------------------------
# Face change detection
# ---------------------------

prev_face_center = None
FACE_CHANGE_THRESHOLD = 25


def face_center(box):

    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return cx, cy


@app.websocket("/liveness")
async def websocket_endpoint(websocket: WebSocket):

    global tracker
    global box
    global frame_count

    global blink_frames
    global liveness_verified
    global liveness_timestamp

    global prev_face_center
    
    global is_registered
    global registered_password
    global registration_embeddings
    global registered_anchor

    await websocket.accept()

    print("Client connected")

    try:

        while True:

            data = await websocket.receive_text()
            
            try:
                import json
                payload = json.loads(data)
                frame_data = payload.get("image", "")
                mode = payload.get("mode", "idle")
                client_password = payload.get("password", "")
            except Exception:
                frame_data = data
                mode = "idle"
                client_password = ""

            frame = decode_base64_frame(frame_data)

            frame = cv2.resize(frame, (320, 240))

            frame_count += 1

            # ---------------------------
            # Detection every 10 frames
            # ---------------------------

            if frame_count % 10 == 0 or tracker is None:

                box = detect_largest_face(frame)

                if box is not None:

                    x1, y1, x2, y2 = box

                    tracker = cv2.TrackerKCF_create()

                    tracker.init(frame, (x1, y1, x2-x1, y2-y1))

            # ---------------------------
            # Tracking step
            # ---------------------------

            elif tracker is not None:

                success, tracked_box = tracker.update(frame)

                if success:

                    x, y, w, h = map(int, tracked_box)

                    box = (x, y, x+w, y+h)

            # ---------------------------
            # Process face
            # ---------------------------

            if box is not None:

                x1, y1, x2, y2 = box

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # ---------------------------
                # Face change detection
                # ---------------------------

                cx, cy = face_center(box)

                if prev_face_center is not None:

                    dist = ((cx-prev_face_center[0])**2 +
                            (cy-prev_face_center[1])**2) ** 0.5

                    if dist > FACE_CHANGE_THRESHOLD:

                        print("Face changed → reset liveness")

                        liveness_verified = False
                        blink_frames = 0

                prev_face_center = (cx, cy)

                # ---------------------------
                # Crop face
                # ---------------------------

                face = crop_face(frame, box)

                if face is not None:
                    
                    # ---------------------------
                    # PyTorch Liveness Check
                    # ---------------------------
                    face_nn = cv2.resize(face, (32, 32))
                    face_nn = cv2.cvtColor(face_nn, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_nn)
                    face_tensor = nn_transform(face_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        preds = liveness_model(face_tensor)
                        probs = F.softmax(preds, dim=1).cpu().numpy()[0]
                        j = np.argmax(probs)
                        label = le.classes_[j]
                    
                    nn_is_real = (label == "real")
                    nn_text = f"NN: {label} ({probs[j]:.2f})"
                    nn_color = (0, 255, 0) if nn_is_real else (0, 0, 255)
                    cv2.putText(frame, nn_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, nn_color, 2)
                    
                    if not nn_is_real:
                        liveness_verified = False

                    face_small = cv2.resize(face, (160,160))

                    landmarks = get_face_landmarks(face_small)

                    if landmarks is not None:

                        EAR = blink_detection(landmarks, face_small.shape)

                        # ---------------------------
                        # Blink detection
                        # ---------------------------

                        if EAR < EAR_THRESHOLD:

                            blink_frames += 1

                        else:

                            if blink_frames >= BLINK_MIN_FRAMES:

                                liveness_verified = True
                                liveness_timestamp = time.time()

                                print("Blink detected → liveness verified")

                            blink_frames = 0

                        cv2.putText(frame, f"EAR:{EAR:.2f}", (10,60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

                # ---------------------------
                # Maintain liveness window
                # ---------------------------

                liveness = False

                if liveness_verified:

                    if time.time() - liveness_timestamp < LIVENESS_TIMEOUT:

                        liveness = True

                    else:

                        liveness_verified = False

                # ---------------------------
                # Debug text
                # ---------------------------

                cv2.putText(frame, f"Liveness:{liveness}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

                # ---------------------------
                # Authentication & Registration Logic
                # ---------------------------

                auth_result = False
                send_password = None
                display_msg = ""
                mode_reset = False

                if mode == "register":
                    if len(registration_embeddings) < REGISTRATION_FRAMES_NEEDED:
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_pil = Image.fromarray(face_rgb)
                        face_tensor = transforms.Compose([
                            transforms.Resize((160, 160)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        ])(face_pil).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            emb = auth_model(face_tensor)
                        registration_embeddings.append(emb)
                        display_msg = f"Extracting Siamese Embeddings... {len(registration_embeddings)}/{REGISTRATION_FRAMES_NEEDED}"
                    else:
                        registered_anchor = torch.cat(registration_embeddings).mean(dim=0, keepdim=True)
                        is_registered = True
                        registered_password = client_password
                        registration_embeddings = []
                        display_msg = "Registration complete! Anchor saved."
                        mode_reset = True
                        
                elif mode == "authenticate":
                    if not is_registered:
                        display_msg = "No user registered yet."
                        mode_reset = True
                    elif not liveness:
                        display_msg = "Liveness not verified yet. Please blink."
                    else:
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_pil = Image.fromarray(face_rgb)
                        face_tensor = transforms.Compose([
                            transforms.Resize((160, 160)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        ])(face_pil).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            test_emb = auth_model(face_tensor)
                            distance = (registered_anchor - test_emb).norm().item()
                        
                        if distance < 1.0:
                            auth_result = True
                            send_password = registered_password
                            display_msg = "Authentication Successful!"
                            mode_reset = True
                        else:
                            auth_result = False
                            display_msg = f"Auth failed (Dist: {distance:.2f})"

                # ---------------------------
                # Send result
                # ---------------------------

                await websocket.send_json({
                    "liveness": liveness,
                    "auth": auth_result,
                    "password": send_password,
                    "message": display_msg,
                    "mode_reset": mode_reset
                })

            # ---------------------------
            # Display debug window
            # ---------------------------

            cv2.imshow("Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception:

        print("Client disconnected")

    finally:

        cv2.destroyAllWindows()


if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )