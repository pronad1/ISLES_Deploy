const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const loadingScreen = document.getElementById('loadingScreen');
const alertError = document.getElementById('alertError');
const resultsContainer = document.getElementById('resultsContainer');
const newScanBtn = document.getElementById('newScanBtn');

let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeUploadZone();
    uploadBtn.addEventListener('click', handleUpload);
    newScanBtn.addEventListener('click', () => location.reload());
});

function initializeUploadZone() {
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
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    selectedFile = file;
    const fileName = file.name;
    const fileSize = (file.size / 1024 / 1024).toFixed(2);

    document.querySelector('.upload-title').innerHTML = '<span style="color: var(--success);">File Selected</span>';
    document.querySelector('.upload-hint').innerHTML = `<strong>${fileName}</strong><br>Size: ${fileSize} MB`;

    uploadBtn.style.display = 'inline-flex';
    fileInput.value = '';
}

async function handleUpload(e) {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    showLoading();

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.message || data.error || 'Upload failed');
        }

        displayResults(data);
    } catch (error) {
        displayError(error.message);
    }
}

function showLoading() {
    uploadZone.parentElement.style.display = 'none';
    alertError.style.display = 'none';
    resultsContainer.style.display = 'none';
    loadingScreen.style.display = 'block';
}

function hideLoading() {
    loadingScreen.style.display = 'none';
}

function displayError(message) {
    hideLoading();
    alertError.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

function displayResults(data) {
    hideLoading();
    resultsContainer.style.display = 'block';

    const lesionPixels = Number(data.lesion_pixels || 0);
    const confidencePct = Number(data.confidence || 0) * 100;

    document.getElementById('infarctVolume').textContent = `${Number(data.infarct_volume_ml || 0).toFixed(2)} mL`;
    document.getElementById('lesionPixels').textContent = `${lesionPixels}`;
    document.getElementById('confidence').textContent = `${confidencePct.toFixed(1)}%`;

    const badge = document.getElementById('statusBadge');
    if (lesionPixels > 0) {
        badge.className = 'status-badge abnormal';
        badge.textContent = 'LESION DETECTED';
    } else {
        badge.className = 'status-badge normal';
        badge.textContent = 'NO LESION DETECTED';
    }

    const stamp = Date.now();
    document.getElementById('previewImage').src = `/${data.image_path}?t=${stamp}`;
    document.getElementById('annotatedImage').src = `/${data.overlay_path}?t=${stamp}`;

    const metadata = data.metadata || {};
    const explanation = data.explanation || {};
    const severity = explanation.severity || {};
    document.getElementById('patientId').textContent = metadata.patient_id || 'N/A';
    document.getElementById('studyDate').textContent = metadata.study_date || 'N/A';
    document.getElementById('modality').textContent = metadata.modality || 'N/A';
    document.getElementById('scanPosition').textContent = metadata.scan_position || explanation.position_note || 'N/A';

    document.getElementById('summaryText').textContent = explanation.summary || 'Result summary is unavailable.';
    document.getElementById('valueGuide').textContent = explanation.value_guide || 'Value guide is unavailable.';
    document.getElementById('reliabilityNote').textContent = explanation.input_reliability || 'Input reliability note is unavailable.';

    const severityChip = document.getElementById('severityChip');
    const severityLevel = severity.level || 'Unknown';
    const severityColor = severity.color || 'neutral';
    const severityNote = severity.note || '';
    severityChip.className = `severity-chip ${severityColor}`;
    severityChip.textContent = severityNote ? `${severityLevel} - ${severityNote}` : severityLevel;

    if (Array.isArray(metadata.shape)) {
        document.getElementById('imageSize').textContent = `${metadata.shape[0]}x${metadata.shape[1]}`;
    } else {
        document.getElementById('imageSize').textContent = 'N/A';
    }

    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
