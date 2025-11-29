import torch
import torchvision.transforms as T
import numpy as np

class TSMClassifier:
    """
    Loads a pretrained TSM model and classifies a sequence of frames.
    """
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.labels = ["No Violence", "Violence"]

    def preprocess(self, frames):
        # frames: list of np.ndarray (H, W, C)
        tensors = [self.transform(f) for f in frames]
        clip = torch.stack(tensors).unsqueeze(0)  # (1, T, C, H, W)
        clip = clip.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
        return clip.to(self.device)

    def classify_clip(self, frames):
        with torch.no_grad():
            clip = self.preprocess(frames)
            logits = self.model(clip)
            pred = torch.argmax(logits, dim=1).item()
            return self.labels[pred]
