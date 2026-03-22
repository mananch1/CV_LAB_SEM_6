import numpy as np
import argparse
import pickle
import time
import cv2
import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from livenessnet import LivenessNet
import imutils

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
    ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
    ap.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detector")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
    if not os.path.isfile(modelPath):
        modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000_fp16.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    print("[INFO] loading liveness detector...")
    le = pickle.loads(open(args["le"], "rb").read())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
    model.load_state_dict(torch.load(args["model"], map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])

    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = vs.read()
        if not ret: break
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                face = frame[startY:endY, startX:endX]
                if face.shape[0] == 0 or face.shape[1] == 0:
                    continue
                    
                face_img = cv2.resize(face, (32, 32))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_img)
                
                face_tensor = transform(face_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    preds = model(face_tensor)
                    probs = F.softmax(preds, dim=1).cpu().numpy()[0]
                    j = np.argmax(probs)
                    label = le.classes_[j]
                
                label_text = f"{label}: {probs[j]:.4f}"
                color = (0, 0, 255) if label == "fake" else (0, 255, 0)
                
                cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.release()
