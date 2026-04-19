// DSANet-ISLES PRO Interaction Engine 

document.addEventListener('DOMContentLoaded', () => {

    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    
    // State Views
    const idleState = document.getElementById('resultsPlaceholder');
    const loadingScreen = document.getElementById('loadingScreen');
    const alertError = document.getElementById('alertError');
    const errorMessage = document.getElementById('errorMessage');
    const resultsContainer = document.getElementById('resultsContainer');
    const metadataContainer = document.getElementById('metadataContainer');

    // Metrics Targets
    const elInfarctVol = document.getElementById('infarctVolume');
    const elLesionPixels = document.getElementById('lesionPixels');
    const elConfidence = document.getElementById('confidenceScore');
    const elPatientId = document.getElementById('metaId');
    const elScanType = document.getElementById('metaMod');
    const elScanDate = document.getElementById('metaDate');
    const elResolution = document.getElementById('metaDim');
    const elClinicalStatus = document.getElementById('clinicalStatus');

    // Image Targeting
    const imgSource = document.getElementById('previewImage');
    const imgOverlay = document.getElementById('annotatedImage');
    const noLesionFlag = document.getElementById('noLesionFlag');

    let currentFile = null;

    // Interaction Hooks
    uploadZone.addEventListener('click', () => fileInput.click());
    
    ['dragover', 'dragenter'].forEach(evt => {
        uploadZone.addEventListener(evt, e => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        uploadZone.addEventListener(evt, e => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        });
    });

    uploadZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', e => handleFiles(e.target.files));

    function handleFiles(files) {
        if (!files.length) return;
        currentFile = files[0];

        const textOutput = uploadZone.querySelector('.upload-text');
        textOutput.textContent = currentFile.name;
        textOutput.classList.add('text-primary');

        uploadBtn.classList.remove('hidden');
        resetViews();
        idleState.classList.remove('hidden');
    }

    uploadBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        if (!currentFile) return;

        resetViews();
        loadingScreen.classList.remove('hidden');
        uploadBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();

            if (res.ok) {
                renderResults(data);
            } else {
                throw new Error(data.error || 'Analysis Pipeline Failed.');
            }
        } catch (err) {
            console.error(err);
            handleError(err.message || "Network Error: Could not reach DSANet Core.");
        } finally {
            uploadBtn.disabled = false;
        }
    });

    function resetViews() {
        idleState.classList.add('hidden');
        loadingScreen.classList.add('hidden');
        alertError.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        metadataContainer.classList.add('hidden');
        noLesionFlag.classList.add('hidden');
        
        imgSource.src = "";
        imgOverlay.src = "";

        if (elClinicalStatus) {
            elClinicalStatus.textContent = 'Processing Complete';
            elClinicalStatus.className = 'status neutral';
        }
    }

    function handleError(msg) {
        resetViews();
        errorMessage.textContent = msg;
        alertError.classList.remove('hidden');
    }

    function renderResults(data) {
        resetViews();
        metadataContainer.classList.remove('hidden');
        resultsContainer.classList.remove('hidden');

        // Render Demographics
        if(data.metadata) {
            elPatientId.textContent = data.metadata.patient_id || '--';
            elScanType.textContent = data.metadata.modality || '--';
            elScanDate.textContent = data.metadata.study_date || '--';
            if (Array.isArray(data.metadata.shape)) {
                elResolution.textContent = `${data.metadata.shape[0]}x${data.metadata.shape[1]}`;
            }
        }

        // Render Volumetrics
        elInfarctVol.textContent = data.infarct_volume_ml ? data.infarct_volume_ml.toFixed(2) : "0.00";
        elLesionPixels.textContent = data.lesion_pixels || "0";
        if (elConfidence) {
            const conf = typeof data.confidence === 'number' ? (data.confidence * 100).toFixed(1) : '0.0';
            elConfidence.textContent = conf;
        }

        const px = parseInt(data.lesion_pixels, 10) || 0;
        if (px === 0) {
            noLesionFlag.classList.remove('hidden');
            if (elClinicalStatus) {
                elClinicalStatus.textContent = 'Normal';
                elClinicalStatus.className = 'status good';
            }
        } else {
            if (elClinicalStatus) {
                elClinicalStatus.textContent = 'Lesion Detected';
                elClinicalStatus.className = 'status warn';
            }
        }

        // Render Viewer 
        const ts = Date.now();
        imgSource.src = `/${data.image_path}?t=${ts}`;
        imgOverlay.src = `/${data.overlay_path}?t=${ts}`;
    }
});