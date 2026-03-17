const video = document.getElementById("video");

async function startCamera() {

    try {

        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = stream;

        document.getElementById("status").innerText = "Camera ready";

    } catch (err) {

        document.getElementById("status").innerText = "Camera access denied";

        console.error(err);
    }

}

startCamera();