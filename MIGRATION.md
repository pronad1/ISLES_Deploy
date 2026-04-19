# ISLES Deployment Migration Notes

This repository has been migrated from a spine lesion classification/detection context to an **ISLES stroke lesion segmentation** context.

## Migration Summary
- Removed Spine-specific assumptions from backend upload flow
- Unified runtime output around segmentation artifacts (`mask`, `overlay`, volumetry)
- Updated repository documentation to ISLES-2022 context
- Kept deployment structure intact (`app.py`, `run.py`, `config/`, `docs/`)

## Current Backend Contract
`POST /upload` now returns:
- `image_path`
- `mask_path`
- `overlay_path`
- `infarct_volume_ml`
- `lesion_pixels`
- `confidence`
- `metadata`

## Operational Notes
- DICOM and non-DICOM image slices are accepted
- Segmentation path is designed for ISLES-style lesion analysis
- Explainability routes are intentionally disabled in this runtime build unless explicitly re-enabled

## Recommended Next Step
Integrate real 3D ISLES model weights (DerNet/SegResNet/Attention U-Net variants) in place of surrogate runtime logic when production checkpoints are available.
