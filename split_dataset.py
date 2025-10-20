import os
import shutil
import random
import sys


data_root = "Processed Image"
output_root = "RoseLeafSet"

# Target class names for output structure (must match existing RoseLeafSet train/val folders)
target_classes = ["Black_Spot", "Dry_Leaf", "Healthy", "Leaf_Hole"]


def map_processed_dir_to_class(dirname: str) -> str | None:
    """Map a processed directory name (e.g. 'Processed_BlackSpot') to one of the target class names.

    Returns the matching target class name or None if no mapping found.
    """
    name = dirname.lower()
    if "black" in name or "spot" in name:
        return "Black_Spot"
    if "dry" in name:
        return "Dry_Leaf"
    if "healthy" in name:
        return "Healthy"
    if "hole" in name or "holes" in name:
        return "Leaf_Hole"
    return None


def main():
    if not os.path.isdir(data_root):
        print(f"Error: data_root '{data_root}' does not exist. Available entries: {os.listdir('.')}")
        sys.exit(2)

    # discover processed folders
    processed_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    if not processed_dirs:
        print(f"No subdirectories found inside '{data_root}'. Please check the processed images location.")
        sys.exit(2)

    # build mapping from processed dirs to target class names
    mapping = {}
    for d in processed_dirs:
        mapped = map_processed_dir_to_class(d)
        if mapped is None:
            print(f"Warning: couldn't map processed dir '{d}' to any target class - skipping")
        else:
            mapping[d] = mapped

    if not mapping:
        print("No processed directories could be mapped to target classes. Exiting.")
        sys.exit(2)

    # prepare output directories
    for split in ["train", "val"]:
        for cls in target_classes:
            os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

    split_ratio = 0.8  # 80% train, 20% val
    random.seed(42)

    for proc_dir, cls in mapping.items():
        src_dir = os.path.join(data_root, proc_dir)
        try:
            imgs = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        except FileNotFoundError:
            print(f"Warning: source directory '{src_dir}' disappeared - skipping")
            continue

        if not imgs:
            print(f"Note: no files found in '{src_dir}' - skipping")
            continue

        random.shuffle(imgs)
        train_count = int(len(imgs) * split_ratio)
        train_imgs = imgs[:train_count]
        val_imgs = imgs[train_count:]

        for img in train_imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(output_root, "train", cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(output_root, "val", cls, img))

        print(f"Processed {len(imgs)} files from '{src_dir}' -> {cls} (train: {len(train_imgs)}, val: {len(val_imgs)})")

    print("\u2705 Dataset structured successfully in 'RoseLeafSet/'")


if __name__ == "__main__":
    main()

