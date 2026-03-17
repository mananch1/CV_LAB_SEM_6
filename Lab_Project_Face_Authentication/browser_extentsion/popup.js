const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

function captureFrame() {

    const video = document.getElementById("video");

    if (video.videoWidth === 0) return;

    canvas.width = 320;
    canvas.height = 240;

    ctx.drawImage(video, 0, 0, 320, 240);

    const frame = canvas.toDataURL("image/jpeg", 0.7);

    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(frame);
    }

}

setInterval(captureFrame, 100);