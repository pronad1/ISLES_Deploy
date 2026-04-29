# ISLES Stroke Lesion Segmentation System

![Python](https://img.shields.io/badge/python-v3.9+-blue)
![Flask](https://img.shields.io/badge/flask-3.0.0-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-red)

This repository now targets **ISLES-2022 style ischemic stroke lesion segmentation** workflows.

## What This Repo Does
- Upload DICOM or standard image slices
- Run an ISLES-style ensemble segmentation pipeline (runtime surrogate)
- Return lesion mask, overlay visualization, lesion pixels, and infarct volume estimate
- Provide metadata and health endpoints for deployment

## Main API
- `POST /upload`: segmentation inference response
- `GET /health`: runtime health and dataset context
- `POST /convert-to-jpeg`: convert DICOM to JPEG

## Expected `/upload` Response Fields
- `image_path`
- `mask_path`
- `overlay_path`
- `infarct_volume_ml`
- `lesion_pixels`
- `confidence`
- `metadata` (`patient_id`, `study_date`, `modality`, `shape`)

## Dataset Context
- Current target dataset: **ISLES-2022**
- Task: **Ischemic Stroke Lesion Segmentation**
- Modality focus: DWI/ADC/FLAIR-aligned lesion analysis workflows

## Run Locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Notebook
Use [isles-2022-dataset-analysis.ipynb](isles-2022-dataset-analysis.ipynb) for dataset EDA and lesion-intensity analysis.
