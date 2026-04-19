// ===== DSANet-ISLES Stroke Segmentation System =====

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const loadingScreen = document.getElementById('loadingScreen');
const alertError = document.getElementById('alertError');
const resultsContainer = document.getElementById('resultsContainer');
let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    // Click to upload
    uploadZone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') {
            fileInput.click();
        }
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileSelect(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });

    uploadBtn.addEventListener('click', handleUpload);
});

function handleFileSelect(file) {
    selectedFile = file;
    const fileName = file.name;
    const fileSize = (file.size / 1024 / 1024).toFixed(2);

    document.querySelector('.upload-title').innerHTML = `<span style="color: var(--success);">✓ File Selected</span>`;
    document.querySelector('.upload-hint').innerHTML = `<strong>${fileName}</strong><br>Size: ${fileSize} MB`;

    uploadBtn.style.display = 'inline-flex';
    fileInput.value = '';
}

async function handleUpload(e) {
    e.preventDefault();
    e.stopPropagation();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    uploadZone.parentElement.style.display = 'none';
    alertError.style.display = 'none';
    resultsContainer.style.display = 'none';
    loadingScreen.style.display = 'block';

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();

        if (!response.ok) throw new Error(data.error || 'Upload failed');

        displayResults(data);
    } catch (error) {
        displayError(error.message);
    }
}

function displayError(message) {
    loadingScreen.style.display = 'none';
    alertError.style.display = 'block';
    document.getElementById('errorMessage').innerHTML = `<strong>❌ Error Processing Slice</strong><br><br>${message}`;
}

function displayResults(data) {
    loadingScreen.style.display = 'none';
    resultsContainer.style.display = 'block';

    // Set Metrics
    if (data.segmentation) {
        document.getElementById('infarctVolume').textContent = data.segmentation.infarct_volume_ml + " mL";
        document.getElementById('lesionPixels').textContent = data.segmentation.lesion_pixels;
    }

    // Set Images
    if (data.images) {
        document.getElementById('previewImage').src = data.images.preview;
        document.getElementById('annotatedImage').src = data.images.annotated;
    }

    // Set Metadata
    if (data.metadata) {
        document.getElementById('patientId').textContent = data.metadata.patient_id;
        document.getElementById('studyDate').textContent = data.metadata.study_date;
        document.getElementById('modality').textContent = data.metadata.modality;
        document.getElementById('imageSize').textContent = data.metadata.dimensions;
    }
}