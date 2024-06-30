class OCRDemo {
    constructor() {
        this.canvas = document.getElementById('canvas');
        if (!this.canvas) {
            console.error('Cannot find canvas element');
            return;
        }
        this.context = this.canvas.getContext('2d');
        this.isDrawing = false;

        this.canvas.addEventListener('mousedown', (event) => this.onMouseDown(event));
        this.canvas.addEventListener('mouseup', () => this.onMouseUp());
        this.canvas.addEventListener('mousemove', (event) => this.onMouseMove(event));

        this.resetCanvas();
    }

    resetCanvas() {
        if (!this.context) {
            console.error('Cannot get canvas context');
            return;
        }
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.context.fillStyle = "white";
        this.context.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Clear the result message
        document.getElementById('result').innerText = '';

        // Clear the digit input field
        document.getElementById('digit').value = '';
    }

    onMouseDown(event) {
        this.isDrawing = true;
        this.context.beginPath();
        this.context.moveTo(event.offsetX, event.offsetY);
    }

    onMouseUp() {
        this.isDrawing = false;
    }

    onMouseMove(event) {
        if (!this.isDrawing) return;
        this.context.lineTo(event.offsetX, event.offsetY);
        this.context.stroke();
    }

    async trainNumber() {
        const digit = document.getElementById('digit').value;
        if (digit === "") {
            alert("Please enter a digit to train.");
            return;
        }

        const dataURL = this.canvas.toDataURL();
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: dataURL,
                label: parseInt(digit)
            })
        });
        const result = await response.json();
        document.getElementById('result').innerText = `Training result: ${result.message}`;
    }

    isCanvasBlank() {
        const blank = document.createElement('canvas');
        blank.width = this.canvas.width;
        blank.height = this.canvas.height;
        const blankCtx = blank.getContext('2d');
        blankCtx.fillStyle = "white";
        blankCtx.fillRect(0, 0, blank.width, blank.height);
        return this.canvas.toDataURL() === blank.toDataURL();
    }

    async predictNumber() {
        if (this.isCanvasBlank()) {
            document.getElementById('result').innerText = 'Please draw something on the canvas.';
            return;
        }

        const dataURL = this.canvas.toDataURL();
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: dataURL })
        });
        const result = await response.json();
        document.getElementById('result').innerText = `Predicted digit: ${result.digit}`;
    }
}

// Initialize OCRDemo after the document has loaded
document.addEventListener('DOMContentLoaded', function() {
    window.ocrDemo = new OCRDemo();
});
