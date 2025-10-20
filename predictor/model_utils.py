import os
from typing import List
import torch
import torchvision.transforms as T
from PIL import Image


class ModelEnsemble:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.classes = ["Black_Spot", "Dry_Leaf", "Healthy", "Leaf_Hole"]
        self.models = self._load_models()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_models(self) -> List[torch.nn.Module]:
        models = []
        # filenames assumed to be in repo root
        candidates = [
            os.path.join(self.model_dir, 'VGG16_roseleaf.pth'),
            os.path.join(self.model_dir, 'ResNet50_roseleaf.pth'),
            os.path.join(self.model_dir, 'DenseNet121_roseleaf.pth'),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            # infer model type from filename
            if 'VGG16' in os.path.basename(path):
                model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
                model.classifier[6] = torch.nn.Linear(4096, 4)
            elif 'ResNet50' in os.path.basename(path):
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
                model.fc = torch.nn.Linear(2048, 4)
            elif 'DenseNet121' in os.path.basename(path):
                model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
                model.classifier = torch.nn.Linear(1024, 4)
            else:
                continue

            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            models.append(model)

        return models

    def predict(self, pil_img: Image.Image, topk: int = 3):
        """Return top-k predictions as a list of (label, probability)."""
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = [torch.softmax(m(x), dim=1) for m in self.models]
        if not outputs:
            return {'error': 'no models available'}
        avg = sum(outputs) / len(outputs)
        probs, preds = torch.topk(avg, k=min(topk, avg.shape[1]), dim=1)
        probs = probs.detach().cpu().numpy()[0]
        preds = preds.detach().cpu().numpy()[0]
        results = []
        for idx, p in zip(preds, probs):
            results.append({'label': self.classes[int(idx)], 'probability': float(p)})
        return {'topk': results}
