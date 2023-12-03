const inputImage = document.getElementById('inputImage');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const canvasOut = document.getElementById('canvas-output');
const ctxOut = canvasOut.getContext('2d');
const resetButton = document.getElementById('resetButton');

let x, y;
let img;
let pipeline_stage = 0;
let thresholdValue = 100, kValue = 5, blurValue = 3;

inputImage.addEventListener('change', handleImageSelect);

function handleImageSelect(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
        resetPipeline();
        img = new Image();
        img.onload = function () {
            const windowWidth = window.innerWidth;
            const windowHeight = window.innerHeight;
            // set canvas size to minimum of image size and screen size
            // canvas.width = Mathemin(img.width, window.innerWidth);
            // canvas.height = Math.min(img.height, window.innerHeight);
            // Resize image to fit canvas in the same aspect ratio
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


resetButton.addEventListener('click', () => {
    resetPipeline();
});

function resetPipeline() {
    pipeline_stage = 0;
    document.getElementById("config-tab-2").style.display = "none";
    document.getElementById("config-tab-1").style.display = "none";
    document.getElementById("config-tab").style.display = "block";
    // Reset out canvas
    ctxOut.clearRect(0, 0, canvasOut.width, canvasOut.height);
}

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
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    let canvasDataUrl;
    if (pipeline_stage == 0) {
        canvasDataUrl = canvas.toDataURL();
    } else if (pipeline_stage >= 1) {
        canvasDataUrl = canvasOut.toDataURL();
    }

    const formData = new FormData();
    formData.append('image', dataURLtoBlob(canvasDataUrl), 'image.png');
    formData.append('width', JSON.stringify(canvas.width));
    formData.append('height', JSON.stringify(canvas.height));

    if (pipeline_stage == 0) {
        pipeline_stage = 1;
        fetch_img(formData, '/process-image-hand');
    } else if (pipeline_stage == 1) {
        document.getElementById("config-tab-1").style.display = "block";
        document.getElementById("config-tab").style.display = "none";
        pipeline_stage = 2;
        fetch_img(formData, '/process-image-hand-1');
    } else if (pipeline_stage == 2) {
        pipeline_stage = 3;
        fetch_img(formData, '/process-image-hand-2');
        document.getElementById("config-tab-1").style.display = "none";
        document.getElementById("config-tab-2").style.display = "block";
    } else if (pipeline_stage == 3) {
        fetch_img(formData, '/process-image-hand-3');
        document.getElementById("config-tab-2").style.display = "none";
    }

});

document.getElementById("reviseButton").addEventListener('click', () => {

    const formData = new FormData();
    formData.append('width', JSON.stringify(canvas.width));
    formData.append('height', JSON.stringify(canvas.height));
    let canvasDataUrl;
    if (pipeline_stage == 1) {
        canvasDataUrl = canvas.toDataURL();
        // Add your JavaScript logic for handling configuration inputs
        var thresholdInput = document.getElementById('threshold');
        var kInput = document.getElementById('k');
        var blurInput = document.getElementById('blur');
        // Access the values and use them as needed all int values
        thresholdValue = parseInt(thresholdInput.value);
        kValue = parseInt(kInput.value);
        blurValue = parseInt(blurInput.value);
        formData.append('image', dataURLtoBlob(canvasDataUrl), 'image.png');
        formData.append('threshold', JSON.stringify(thresholdValue));
        formData.append('k', JSON.stringify(kValue));
        formData.append('blur', JSON.stringify(blurValue));
        fetch_img(formData, '/process-image-hand');
    } else if (pipeline_stage == 2) {
        canvasDataUrl = canvasOut.toDataURL();
        let lineThreshold = parseInt(document.getElementById('line_threshold').value);
        let lineMinLineLength = parseFloat(document.getElementById('min_line_percentage').value);
        let lineMaxLineGap = parseInt(document.getElementById('max_line_gap').value);
        formData.append('image', dataURLtoBlob(canvasDataUrl), 'image.png');
        formData.append('line_threshold', JSON.stringify(lineThreshold));
        formData.append('min_line_percent', JSON.stringify(lineMinLineLength));
        formData.append('max_line_gap', JSON.stringify(lineMaxLineGap));
        fetch_img(formData, '/process-image-hand-1');
    } else if (pipeline_stage == 3) {
        canvasDataUrl = canvasOut.toDataURL();
        let pad_size_x = parseInt(document.getElementById('pad_size_x').value);
        let pad_size_y = parseInt(document.getElementById('pad_size_y').value);
        formData.append('image', dataURLtoBlob(canvasDataUrl), 'image.png');
        formData.append('pad_size_x', JSON.stringify(pad_size_x));
        formData.append('pad_size_y', JSON.stringify(pad_size_y));
        fetch_img(formData, '/process-image-hand-2');
    }
});


function fetch_img(formData, url) {
    fetch(url, {
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

// 