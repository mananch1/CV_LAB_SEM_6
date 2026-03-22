import cv2
import pickle
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
from livenessnet import LivenessNet

print("[INFO] Loading label encoder and model...")
le = pickle.loads(open("le.pickle", "rb").read())
device = torch.device("cpu")
model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.load_state_dict(torch.load("liveness.model", map_location=device))
model.eval()

print("[INFO] Loading face detector...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

print("[INFO] Reading test.jpg...")
frame = cv2.imread("test.jpg")
if frame is None:
    print("[ERROR] Could not read test.jpg")
    exit(1)

(h, w) = frame.shape[:2]
target_width = 600
ratio = target_width / float(w)
dim = (target_width, int(h * ratio))
frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
(h, w) = frame.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

found = False
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # bound
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w, endX), min(h, endY)
        face = frame[startY:endY, startX:endX]
        if face.shape[0] == 0 or face.shape[1] == 0: continue
        
        face_img = cv2.resize(face, (32, 32))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_img)
        transform = transforms.Compose([transforms.ToTensor()])
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(face_tensor)
            probs = F.softmax(preds, dim=1).cpu().numpy()[0]
            j = np.argmax(probs)
            label = le.classes_[j]
            found = True
            print(f"[RESULT] Detected face: {label} with confidence {probs[j]:.4f}")

if not found:
    print("[INFO] No faces found in test.jpg")
