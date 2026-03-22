import cv2
import pickle
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import imutils
import argparse
from livenessnet import LivenessNet
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to optional input video/image (leave blank for webcam)")
args = vars(ap.parse_args())

print("[INFO] Loading label encoder and model...")
le = pickle.loads(open("le.pickle", "rb").read())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.load_state_dict(torch.load("liveness.model", map_location=device))
model.to(device)
model.eval()

print("[INFO] Loading face detector...")
# Using the fp16 version which we verified exists
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
transform = transforms.Compose([transforms.ToTensor()])

input_path = args["input"]
if input_path == "":
    print("[INFO] Starting webcam stream... Press 'q' to quit.")
    vs = cv2.VideoCapture(0)
    is_video = True
elif input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
    print(f"[INFO] Reading image: {input_path}. Press any key to close the window.")
    vs = None
    is_video = False
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[ERROR] Could not load image: {input_path}")
        exit(1)
else:
    print(f"[INFO] Opening video file: {input_path}... Press 'q' to quit.")
    vs = cv2.VideoCapture(input_path)
    is_video = True

def process_frame(frame):
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure bounding boxes fall within the dimensions of the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            face = frame[startY:endY, startX:endX]
            if face.shape[0] == 0 or face.shape[1] == 0: continue

            # Preprocess the face for the liveness model
            face_img = cv2.resize(face, (32, 32))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_img)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Pass the face through the model to determine liveness
            with torch.no_grad():
                preds = model(face_tensor)
                probs = F.softmax(preds, dim=1).cpu().numpy()[0]
                j = np.argmax(probs)
                label = le.classes_[j]

            # Draw the label and bounding box on the frame
            label_text = f"{label}: {probs[j]:.4f}"
            color = (0, 0, 255) if label == "fake" else (0, 255, 0)

            cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
    return frame

if not is_video:
    output_frame = process_frame(frame)
    cv2.imshow("Liveness Test", output_frame)
    cv2.waitKey(0)
else:
    while True:
        ret, frame = vs.read()
        if not ret: break
        
        output_frame = process_frame(frame)
        cv2.imshow("Liveness Test", output_frame)
        
        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.release()

cv2.destroyAllWindows()
