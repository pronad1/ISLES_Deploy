# Contributing to ISLES Stroke Segmentation System

Thank you for contributing.

## How to Contribute
1. Open an issue with a clear problem statement.
2. Include repro steps, expected behavior, and environment details.
3. Submit a pull request with focused, reviewable changes.

## Development Focus Areas
- ISLES data ingestion and preprocessing
- Segmentation quality and volumetric metrics
- Clinical visualization quality
- Runtime stability and deployment reliability
- Documentation and reproducibility

## Code Standards
- Follow PEP 8
- Keep functions small and testable
- Add concise comments only where needed
- Update docs when behavior changes

## Validation Checklist
- Application starts (`python app.py`)
- `/health` returns healthy status
- `/upload` returns valid segmentation payload
- Notebook cells run without import/runtime errors in a configured environment

## Community Conduct
- Be respectful and constructive.
- Keep discussions evidence-based and reproducible.
