# ISLES Methodology Overview

## Dataset
This project is aligned to **ISLES-2022** ischemic stroke lesion segmentation workflows.

## Core Objective
Segment ischemic lesion regions from stroke imaging slices and produce:
- lesion mask
- lesion overlay
- pixel-level lesion burden
- estimated infarct volume

## Runtime Inference Strategy
The deployed runtime currently uses a lightweight ISLES-style ensemble surrogate:
1. Input normalization to 0-255
2. 2D-to-3D volume expansion (64 slices)
3. Test-time augmentation (flip-based)
4. Weighted ensemble fusion (0.4 / 0.3 / 0.3)
5. Binary lesion mask generation
6. Overlay and volumetry output

## Output Contract
`POST /upload` returns:
- `image_path`
- `mask_path`
- `overlay_path`
- `infarct_volume_ml`
- `lesion_pixels`
- `confidence`
- metadata (`patient_id`, `study_date`, `modality`, `shape`)

## Clinical Notes
- This system is for research and educational use.
- Final diagnosis should be made by licensed medical professionals.
