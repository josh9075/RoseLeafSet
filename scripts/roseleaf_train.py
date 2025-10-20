import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("RoseLeafSet/train", transform=train_transform)
val_dataset = datasets.ImageFolder("RoseLeafSet/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

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

def train_model(model, train_loader, val_loader, model_name, epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[{model_name}] Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        # Validation
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
        print(f"[{model_name}] Val Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), f"{model_name}_roseleaf.pth")
    print(f"{model_name} model saved!")

if __name__ == '__main__':
    vgg16 = build_model("vgg16")
    resnet50 = build_model("resnet50")
    densenet121 = build_model("densenet121")
    train_model(vgg16, train_loader, val_loader, "VGG16", epochs=5)
    train_model(resnet50, train_loader, val_loader, "ResNet50", epochs=5)
    train_model(densenet121, train_loader, val_loader, "DenseNet121", epochs=5)
