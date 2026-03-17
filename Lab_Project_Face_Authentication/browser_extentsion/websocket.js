let socket;

function startSocket() {

    socket = new WebSocket("ws://localhost:8000/liveness");

    socket.onopen = () => {
        console.log("WebSocket connected");
    };

    socket.onmessage = (event) => {

        const data = JSON.parse(event.data);

        if (data.liveness) {
            document.getElementById("status").innerText = "Liveness confirmed";
        }

        if (data.auth) {
            document.getElementById("status").innerText = "Authentication success";
        }
    };

}

startSocket();