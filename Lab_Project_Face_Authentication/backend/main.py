import cv2
import uvicorn
from fastapi import FastAPI, WebSocket
from utils import *

app = FastAPI()

blink_frames = 0
blink_detected = False

EAR_THRESHOLD = 0.21
MOTION_THRESHOLD = 1.0
HEAD_THRESHOLD = 5

@app.websocket("/liveness")
async def websocket_endpoint(websocket: WebSocket):
    global blink_frames
    global blink_detected

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            frame = decode_base64_frame(data)
            frame = cv2.resize(frame, (320, 240))
            box = detect_largest_face(frame)

            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                face = crop_face(frame, box)

                if face is not None:
                    landmarks = get_face_landmarks(face)

                    if landmarks is not None:
                        # --- FIXED THE CALLS HERE ---
                        EAR = blink_detection(landmarks, face.shape)
                        yaw = head_pose_estimation(landmarks, face.shape)
                        motion = optical_flow_motion(face)

                        # Blink logic
                        if EAR is not None:
                            if EAR < EAR_THRESHOLD:
                                blink_frames += 1
                            else:
                                if blink_frames >= 3:
                                    blink_detected = True
                                blink_frames = 0

                        # Scoring logic
                        score = 0
                        if blink_detected:
                            score += 1

                        if yaw is not None and abs(yaw) > HEAD_THRESHOLD:
                            score += 1

                        if motion > MOTION_THRESHOLD:
                            score += 1

                        liveness = score >= 2

                        # Draw info on frame
                        cv2.putText(frame, f"Score: {score}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Motion: {motion:.2f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        if EAR is not None:
                            cv2.putText(frame, f"EAR: {EAR:.2f}", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Send results back
                        await websocket.send_json({
                            "liveness": liveness,
                            "score": score,
                            "auth": False
                        })
                    else:
                        EAR = None

            # Optional: Display the frame
            # WARNING: cv2.imshow can sometimes cause hanging in async FastAPI loops 
            # if not handled in a separate thread, but is fine for quick local testing.
            cv2.imshow("Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"WebSocket connection closed or errored: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )