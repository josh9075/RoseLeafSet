This folder contains runnable scripts for preprocessing and training.

- `split_dataset.py` - splits processed images into `RoseLeafSet/train` and `RoseLeafSet/val`.
- `roseleaf_train.py` - training script for VGG16, ResNet50, DenseNet121.

Usage (from project root):
```
python scripts/split_dataset.py
python scripts/roseleaf_train.py
```

Place large model checkpoints in the `models/` folder (not tracked by git).
