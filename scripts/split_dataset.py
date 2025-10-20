import os, shutil, random

data_root = "Processed Image"
output_root = "RoseLeafSet"

target_classes = ["Black_Spot", "Dry_Leaf", "Healthy", "Leaf_Hole"]

def map_processed_dir_to_class(dirname: str) -> str | None:
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
    os.makedirs(output_root, exist_ok=True)
    processed_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    mapping = {}
    for d in processed_dirs:
        mapped = map_processed_dir_to_class(d)
        if mapped:
            mapping[d] = mapped

    for split in ["train", "val"]:
        for cls in target_classes:
            os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

    split_ratio = 0.8
    random.seed(42)
    for proc_dir, cls in mapping.items():
        src_dir = os.path.join(data_root, proc_dir)
        imgs = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        random.shuffle(imgs)
        train_count = int(len(imgs) * split_ratio)
        train_imgs = imgs[:train_count]
        val_imgs = imgs[train_count:]
        for img in train_imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(output_root, "train", cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(output_root, "val", cls, img))

    print("Dataset structured successfully in 'RoseLeafSet/'")

if __name__ == '__main__':
    main()
