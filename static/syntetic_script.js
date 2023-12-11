const inputImage = document.getElementById('inputImage');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const canvasOut = document.getElementById('canvas-output');
const ctxOut = canvasOut.getContext('2d');
const resetButton = document.getElementById('resetButton');
const bBox1 = document.getElementById('boundingBox1Button');
const bBox2 = document.getElementById('boundingBox2Button');
const bBox3 = document.getElementById('boundingBox3Button');

let startX, startY, isDrawing = false;
let x, y;
let img;
let selectedBoundingBox = 'Graph';
let graphBoundingBox, axisYBoundingBox, axisXBoundingBox;
let original_image_blob;
let isXAxisRotated = false;

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
    resetBoundingBoxes();
    original_image_blob = file;

    reader.onload = function (e) {
        img = new Image();
        img.onload = function () {
            const windowWidth = window.innerWidth;
            const windowHeight = window.innerHeight;
            const img_ration = img.width / img.height;
            const max_percent = 0.5;
            if (img.width > windowWidth * max_percent) {
                img.width = windowWidth * max_percent;
                img.height = img.width / img_ration;
            } else if (img.height > windowHeight * max_percent) {
                img.height = windowHeight * max_percent;
                img.width = img.height * img_ration;
            }
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

resetButton.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    resetBoundingBoxes();
});

bBox1.addEventListener('click', () => {
    selectedBoundingBox = 'Graph';
});

bBox2.addEventListener('click', () => {
    selectedBoundingBox = 'AxisY';
});

bBox3.addEventListener('click', () => {
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
    if ((graphBoundingBox && axisXBoundingBox && axisYBoundingBox)) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        const formData = new FormData();
        formData.append('image', original_image_blob, 'image.png');
        formData.append('graphBox', JSON.stringify(graphBoundingBox));
        formData.append('axisXBox', JSON.stringify(axisXBoundingBox));
        formData.append('axisYBox', JSON.stringify(axisYBoundingBox));
        formData.append('isXAxisRotated', JSON.stringify(isXAxisRotated));
        let plotType = document.getElementById('plotType').value;
        formData.append('plotType', JSON.stringify(plotType));

        // resetBoundingBoxes();

        fetch('/process-image-syntetic', {
            method: 'POST',
            body: formData
        })
        .then(response => response.blob())
        .then(blob => {
            // Create a link element and trigger the download
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'data.csv';
            a.textContent = 'Download CSV';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        })
        .catch(error => console.error(error));
    } else {
        alert('Please select all 3 bounding boxes');
    }
});

function resetBoundingBoxes() {
    graphBoundingBox = null;
    axisXBoundingBox = null;
    axisYBoundingBox = null;
}

function openNav() {
    document.getElementById("sidebar").style.width = "250px";
    document.getElementById("main").style.marginLeft = "250px";
    document.getElementById("openConfig").style.display = "none";
}

function closeNav() {
    document.getElementById("sidebar").style.width = "0";
    document.getElementById("main").style.marginLeft = "0";
    document.getElementById("openConfig").style.display = "block";
}

// Close Nav on esq key
document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
        closeNav();
    }
    if (event.key === 'o') {
        openNav();
    }
});

// JavaScript code to toggle between input and output canvases
var toggleSwitch = document.getElementById('toggleSwitch');

toggleSwitch.addEventListener('change', function () {
    if (toggleSwitch.checked) {
        isXAxisRotated = true;
        console.log('checked');
    } else {
        isXAxisRotated = false;
        console.log('unchecked');
    }
});