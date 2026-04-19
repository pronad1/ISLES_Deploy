# Deployment Guide (ISLES Segmentation)

## Local Run
```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000`.

## Runtime Endpoints
- `GET /health`
- `POST /upload`
- `POST /convert-to-jpeg`

## Upload Inputs
- `.dcm`, `.dicom`, `.jpg`, `.jpeg`, `.png`
- Max file size: 16MB

## Expected `/upload` Outputs
- Original image path
- Segmentation mask path
- Overlay path
- Lesion pixels
- Infarct volume estimate (mL)
- Confidence
- Metadata

## Docker
```bash
docker build -t isles-segmentation .
docker run -p 5000:5000 isles-segmentation
```

## Health Check
```bash
curl http://localhost:5000/health
```

## Troubleshooting
- If upload fails, confirm file extension and readability.
- If DICOM conversion fails, verify valid pixel data in the DICOM object.
- If memory is constrained, keep batch usage to single-image requests.
