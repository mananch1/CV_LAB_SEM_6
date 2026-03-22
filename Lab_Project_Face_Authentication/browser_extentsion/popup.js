const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

window.currentMode = "idle";

function captureFrame() {

    const video = document.getElementById("video");

    if (video.videoWidth === 0) return;

    canvas.width = 320;
    canvas.height = 240;

    ctx.drawImage(video, 0, 0, 320, 240);

    const frame = canvas.toDataURL("image/jpeg", 0.7);

    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            image: frame,
            mode: window.currentMode,
            password: document.getElementById("password").value
        }));
    }

}

setInterval(captureFrame, 100);

document.getElementById("registerBtn").addEventListener("click", () => {
    window.currentMode = "register";
    document.getElementById("status").innerText = "Registering...";
});

document.getElementById("accessBtn").addEventListener("click", () => {
    window.currentMode = "authenticate";
    document.getElementById("status").innerText = "Authenticating...";
});