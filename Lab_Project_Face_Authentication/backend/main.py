from fastapi import FastAPI, WebSocket
import cv2
import uvicorn

from utils import decode_base64_frame, detect_largest_face, crop_face

app = FastAPI()


@app.websocket("/liveness")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    while True:

        data = await websocket.receive_text()

        frame = decode_base64_frame(data)

        box = detect_largest_face(frame)

        if box is not None:

            x1,y1,x2,y2 = box

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            face = crop_face(frame, box)

            if face is not None:
                cv2.imshow("Face Crop", face)

        cv2.imshow("Incoming Stream", frame)

        cv2.waitKey(1)

        await websocket.send_json({
            "liveness": True,
            "auth": False
        })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)