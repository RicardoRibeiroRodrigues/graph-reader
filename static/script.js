const inputImage = document.getElementById('inputImage');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const canvasOut = document.getElementById('canvas-output');
const ctxOut = canvasOut.getContext('2d');

let startX, startY, isDrawing = false;
let x, y;
let img;
let selectedBoundingBox = 'Graph';
let graphBoundingBox, axisYBoundingBox, axisXBoundingBox;

class BoundingBox {
    constructor(x_min, y_min, width, height) {
        this.x_min = x_min;
        this.y_min = y_min;
        this.width = width;
        this.height = height;
    }
}

inputImage.addEventListener('change', handleImageSelect);

function handleImageSelect(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    // reset bounding boxes
    graphBoundingBox = null;
    axisYBoundingBox = null;
    axisXBoundingBox = null;

    reader.onload = function (e) {
        img = new Image();
        img.onload = function () {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = e.target.result;
    };

    reader.readAsDataURL(file);
}

canvas.addEventListener('mousedown', (event) => {
    isDrawing = true;
    startX = event.clientX - canvas.getBoundingClientRect().left;
    startY = event.clientY - canvas.getBoundingClientRect().top;
});

canvas.addEventListener('mousemove', (event) => {
    if (!isDrawing) return;

    x = event.clientX - canvas.getBoundingClientRect().left;
    y = event.clientY - canvas.getBoundingClientRect().top;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    let bound = new BoundingBox(startX, startY, x - startX, y - startY);
    if (selectedBoundingBox === 'Graph') {
        graphBoundingBox = bound;
    } else if (selectedBoundingBox === 'AxisY') {
        axisYBoundingBox = bound;
    } else if (selectedBoundingBox === 'AxisX') {
        axisXBoundingBox = bound;
    }

    if (graphBoundingBox) {
        drawBoundingBox(graphBoundingBox, 'red');
    }
    if (axisYBoundingBox) {
        drawBoundingBox(axisYBoundingBox, 'green');
    }
    if (axisXBoundingBox) {
        drawBoundingBox(axisXBoundingBox, 'blue');
    }
});

canvas.addEventListener('mouseup', () => {
    if (isDrawing) {
        isDrawing = false;
        const width = x - startX;
        const height = y - startY;
        console.log(`x: ${startX}, y: ${startY}, max_x: ${width}, max_y: ${height}`);
    }
});

function drawBoundingBox(boundingBox, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(boundingBox.x_min, boundingBox.y_min, boundingBox.width, boundingBox.height);
}

document.getElementById('resetButton').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    axisXBoundingBox = null;
    axisYBoundingBox = null;
    graphBoundingBox = null;
});

document.getElementById('boundingBox1Button').addEventListener('click', () => {
    selectedBoundingBox = 'Graph';
});

document.getElementById('boundingBox2Button').addEventListener('click', () => {
    selectedBoundingBox = 'AxisY';
});

document.getElementById('boundingBox3Button').addEventListener('click', () => {
    selectedBoundingBox = 'AxisX';
});

function dataURLtoBlob(dataURL) {
    const parts = dataURL.split(';base64,');
    const contentType = parts[0].split(':')[1];
    const raw = window.atob(parts[1]);
    const rawLength = raw.length;
    const uInt8Array = new Uint8Array(rawLength);

    for (let i = 0; i < rawLength; ++i) {
        uInt8Array[i] = raw.charCodeAt(i);
    }

    return new Blob([uInt8Array], { type: contentType });
}

document.getElementById('sendButton').addEventListener('click', () => {
    if (graphBoundingBox && axisXBoundingBox && axisYBoundingBox) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        const canvasDataUrl = canvas.toDataURL(); 

        const formData = new FormData();
        formData.append('image', dataURLtoBlob(canvasDataUrl), 'image.png');
        formData.append('graphBox', JSON.stringify(graphBoundingBox));
        formData.append('axisXBox', JSON.stringify(axisXBoundingBox));
        formData.append('axisYBox', JSON.stringify(axisYBoundingBox));
        formData.append('width', JSON.stringify(canvas.width));
        formData.append('height', JSON.stringify(canvas.height));


        fetch('/process-image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.blob())
        .then(blob => {
            const objectURL = URL.createObjectURL(blob);

            const image = new Image();
            image.onload = function () {
                canvasOut.width = image.width;
                canvasOut.height = image.height;
                ctxOut.drawImage(image, 0, 0, canvasOut.width, canvasOut.height);
            };
            image.src = objectURL;
        })
        .catch(error => console.error(error));
    } else {
        alert('Please select all 3 bounding boxes');
    }
});