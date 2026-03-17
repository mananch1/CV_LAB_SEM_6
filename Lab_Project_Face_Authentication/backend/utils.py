import base64
import cv2
import numpy as np

# Load OpenCV DNN face detector
net = cv2.dnn.readNetFromCaffe(
    "../deploy.prototxt",
    "../res10_300x300_ssd_iter_140000_fp16.caffemodel"
)


def decode_base64_frame(data):
    """
    Convert base64 image string to OpenCV frame
    """

    header, encoded = data.split(",", 1)

    img_bytes = base64.b64decode(encoded)

    np_arr = np.frombuffer(img_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return frame


def detect_largest_face(frame):

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104, 177, 123)
    )

    net.setInput(blob)

    detections = net.forward()

    largest_face = None
    largest_area = 0

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])

            (x1,y1,x2,y2) = box.astype("int")

            pad = int(0.15 * (x2 - x1))

            x1 = max(0, x1-pad)
            y1 = max(0, y1-pad)
            x2 = min(w, x2+pad)
            y2 = min(h, y2+pad)

            area = (x2-x1) * (y2-y1)

            if area > largest_area:
                largest_area = area
                largest_face = (x1,y1,x2,y2)

    return largest_face


def crop_face(frame, box):

    if box is None:
        return None

    x1,y1,x2,y2 = box

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.resize(face, (160,160))

    return face