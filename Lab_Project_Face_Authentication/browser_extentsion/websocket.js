let socket;

function startSocket() {

    socket = new WebSocket("ws://localhost:8000/liveness");

    socket.onopen = () => {
        console.log("WebSocket connected");
    };

    socket.onmessage = (event) => {

        const data = JSON.parse(event.data);

        if (data.message) {
            document.getElementById("status").innerText = data.message;
        } else if (data.liveness) {
            document.getElementById("status").innerText = "Liveness confirmed. " + (window.currentMode !== 'idle' ? 'Processing...' : '');
        }

        if (data.mode_reset) {
            window.currentMode = "idle";
        }

        if (data.auth === true && data.password) {
            document.getElementById("status").innerText = "Authentication success!";
            document.getElementById("secretDisplay").innerText = "Secret Password: " + data.password;
            window.currentMode = "idle";
        } else if (data.auth === false && window.currentMode === "authenticate") {
            document.getElementById("secretDisplay").innerText = "Access Denied.";
        }
    };

}

startSocket();