# RoseLeafSet

This repository contains scripts and a small Django webapp for training and serving a leaf disease classifier.

Repository layout
- `scripts/` - preprocessing and training scripts
	- `split_dataset.py` - split processed images into `RoseLeafSet/train` and `RoseLeafSet/val`
	- `roseleaf_train.py` - training script for VGG16 / ResNet50 / DenseNet121
- `models/` - (not tracked) place model checkpoint files here, e.g. `VGG16_roseleaf.pth`
- `predictor/` + `webapp/` - Django app and project that serves an upload UI and API
- `requirements.txt` - Python dependencies
- `docs/` - minimal GitHub Pages demo site

Quick start (development)
1. Create & activate a virtualenv (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place your trained model checkpoint files inside `models/` (same names as used by training outputs: `VGG16_roseleaf.pth`, `ResNet50_roseleaf.pth`, `DenseNet121_roseleaf.pth`).

3. Run the Django dev server and open the UI:
```bash
.venv/bin/python manage.py runserver
# then open http://127.0.0.1:8000/
```

4. Run preprocessing or training scripts:
```bash
python scripts/split_dataset.py
python scripts/roseleaf_train.py
```

Notes
- Large data & model files are ignored by `.gitignore`. Do not commit model checkpoints to the repo unless necessary.
- For production deployment, containerize the app (Docker) and use gunicorn + nginx or a managed service.
