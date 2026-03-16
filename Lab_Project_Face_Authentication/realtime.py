import cv2 as cv
import numpy as np

# Load face detection model
net = cv.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Open webcam
cap = cv.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Create blob
    blob = cv.dnn.blobFromImage(
        cv.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104, 177, 123)
    )

    net.setInput(blob)
    detections = net.forward()

    largest_face = None
    largest_area = 0

    # Detect faces
    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x1,y1,x2,y2) = box.astype("int")

            # Padding (15% of face width)
            pad = int(0.15 * (x2-x1))

            x1 = max(0, x1-pad)
            y1 = max(0, y1-pad)
            x2 = min(w, x2+pad)
            y2 = min(h, y2+pad)

            area = (x2-x1) * (y2-y1)

            if area > largest_area:
                largest_area = area
                largest_face = (x1,y1,x2,y2)

    # Process largest face only
    if largest_face is not None:

        x1,y1,x2,y2 = largest_face

        # Draw bounding box
        cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # Crop face
        face = frame[y1:y2, x1:x2]

        if face.size > 0:

            face = cv.resize(face,(160,160))
            cv.imshow("Face Crop", face)

    # Show webcam feed
    cv.imshow("Webcam", frame)

    # Quit with q
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()