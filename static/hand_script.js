const inputImage = document.getElementById('inputImage');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const canvasOut = document.getElementById('canvas-output');
const ctxOut = canvasOut.getContext('2d');
const resetButton = document.getElementById('resetButton');

let x, y;
let img;
let original_image_blob;
let threshold_image;
let corrected_image;
let pipeline_stage = 0;
let thresholdValue = 100, kValue = 5, blurValue = 3;

inputImage.addEventListener('change', handleImageSelect);

function handleImageSelect(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    original_image_blob = file;

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
    const formData = new FormData();
    
    if (pipeline_stage == 0) {
        pipeline_stage = 1;
        formData.append('image', original_image_blob, 'image.png');
        fetch_img(formData, '/process-image-hand');
    } else if (pipeline_stage == 1) {
        document.getElementById("config-tab-1").style.display = "block";
        document.getElementById("config-tab").style.display = "none";
        pipeline_stage = 2;
        formData.append('th_image', threshold_image, 'image.png');
        formData.append('image', original_image_blob, 'image1.png');
        fetch_img(formData, '/process-image-hand-1');
    } else if (pipeline_stage == 2) {
        pipeline_stage = 3;
        formData.append('image', original_image_blob, 'image.png');
        formData.append('th_image', threshold_image, 'image1.png');
        fetch_img(formData, '/process-image-hand-2');
        document.getElementById("config-tab-1").style.display = "none";
        document.getElementById("config-tab-2").style.display = "block";
    } else if (pipeline_stage == 3) {
        document.getElementById("config-tab-2").style.display = "none";
        formData.append('image', corrected_image, 'image.png');
        let plotType = document.getElementById('plotType').value;
        formData.append('plotType', JSON.stringify(plotType));

        fetch('/process-image-hand-3', {
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
    }

});

document.getElementById("reviseButton").addEventListener('click', () => {

    const formData = new FormData();

    if (pipeline_stage == 1) {
        // Revise options for image thresholding
        var thresholdInput = document.getElementById('threshold');
        var kInput = document.getElementById('k');
        var blurInput = document.getElementById('blur');
        // Access the values and use them as needed all int values
        thresholdValue = parseInt(thresholdInput.value);
        kValue = parseInt(kInput.value);
        blurValue = parseInt(blurInput.value);
        // Send both the original image
        formData.append('image', original_image_blob, 'image.png');
        // Send configuration parameters
        formData.append('threshold', JSON.stringify(thresholdValue));
        formData.append('k', JSON.stringify(kValue));
        formData.append('blur', JSON.stringify(blurValue));
        fetch_img(formData, '/process-image-hand');
    } else if (pipeline_stage == 2) {
        let lineThreshold = parseInt(document.getElementById('line_threshold').value);
        let lineMinLineLength = parseFloat(document.getElementById('min_line_percentage').value);
        let lineMaxLineGap = parseInt(document.getElementById('max_line_gap').value);
        // Send both the original image and the thresholded image
        formData.append('th_image', threshold_image, 'image.png');
        formData.append('image', original_image_blob, 'image1.png');
        // Send configuration parameters
        formData.append('line_threshold', JSON.stringify(lineThreshold));
        formData.append('min_line_percent', JSON.stringify(lineMinLineLength));
        formData.append('max_line_gap', JSON.stringify(lineMaxLineGap));
        fetch_img(formData, '/process-image-hand-1');
    } else if (pipeline_stage == 3) {
        let pad_size_x = parseFloat(document.getElementById('pad_size_x').value);
        let pad_size_y = parseFloat(document.getElementById('pad_size_y').value);
        // Send both the original image and the thresholded image
        formData.append('image', original_image_blob, 'image.png');
        formData.append('th_image', threshold_image, 'image1.png');
        // Send configuration parameters
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

                const img_ration = image.width / image.height;
                const max_percent = 0.5;
                if (image.width > window.innerWidth * max_percent) {
                    image.width = window.innerWidth * max_percent;
                    image.height = image.width / img_ration;
                }
                else if (image.height > window.innerHeight * max_percent) {
                    image.height = window.innerHeight * max_percent;
                    image.width = image.height * img_ration;
                }
                canvasOut.width = image.width;
                canvasOut.height = image.height;
                ctxOut.drawImage(image, 0, 0, canvasOut.width, canvasOut.height);
            };
            image.src = objectURL;
            if (pipeline_stage == 1) {
                threshold_image = blob;
            } else if (pipeline_stage == 3) {
                corrected_image = blob;
            } 
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
