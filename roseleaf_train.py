# ===========================
# RoseLeafSet Classification Training
# Models: VGG16, ResNet50, DenseNet121
# Author: Md. Tareque Jamil Josh
# ===========================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader
except Exception as e:
    print("Missing required PyTorch / torchvision packages. Please install them before running this script.")
    print("If you're using pyenv, activate your environment and run: pip install -r requirements.txt")
    raise

try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None

try:
    from tqdm import tqdm
except Exception:
    # fallback simple progress
    def tqdm(x, **kwargs):
        return x

import numpy as np

# ---------------------------
# Step 1: Device Setup (MPS for Mac M2)
# ---------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ---------------------------
# Step 2: Preprocessing & Dataloaders
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("RoseLeafSet/train", transform=train_transform)
val_dataset = datasets.ImageFolder("RoseLeafSet/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Classes:", train_dataset.classes)
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))

# ---------------------------
# Step 3: Model Builder
# ---------------------------
def build_model(model_name):
    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 4)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 4)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, 4)
    else:
        raise ValueError("Invalid model name")
    return model.to(device)

# ---------------------------
# Step 4: Training Function
# ---------------------------
def train_model(model, train_loader, val_loader, model_name, epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[{model_name}] Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        # Validation phase
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"[{model_name}] Val Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

    torch.save(model.state_dict(), f"{model_name}_roseleaf.pth")
    print(f"✅ {model_name} model saved!\n")
    return model

# ---------------------------
# Step 5: Train Each Model
# ---------------------------
vgg16 = build_model("vgg16")
resnet50 = build_model("resnet50")
densenet121 = build_model("densenet121")

vgg16 = train_model(vgg16, train_loader, val_loader, "VGG16", epochs=5)
resnet50 = train_model(resnet50, train_loader, val_loader, "ResNet50", epochs=5)
densenet121 = train_model(densenet121, train_loader, val_loader, "DenseNet121", epochs=5)

# ---------------------------
# Step 6: Ensemble Evaluation
# ---------------------------
def ensemble_predict(models, dataloader):
    for model in models:
        model.to(device)
        model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Ensemble Evaluating"):
            images = images.to(device)
            outputs = [torch.softmax(model(images), dim=1) for model in models]
            avg_output = sum(outputs) / len(models)
            _, preds = torch.max(avg_output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n🌿 Ensemble Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    return acc, f1

ensemble_predict([vgg16, resnet50, densenet121], val_loader)
