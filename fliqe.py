from collections import deque
import cv2
import numpy as np
from models import DistortionBinaryClassifier, IQAEncoder
import torch
from torchvision import transforms


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # 224 or 384
    transforms.ToTensor(),
])


class FLIQE:
    def __init__(self, encoder_model_path='models/resnet50_128_out.pth', quality_model_path='models/best_binary_classifier.pth', smoothing_window=300, threshold=0.5):
        encoder_model = IQAEncoder(feature_dim=128, model_name='resnet50').to(device)
        encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=device))
        encoder_model.eval()
        self.quality_model = DistortionBinaryClassifier(iqa_encoder=encoder_model).to(device)
        self.quality_model.load_state_dict(torch.load(quality_model_path, map_location=device))
        self.quality_model.eval()
        self.smoothing_window = smoothing_window
        self.threshold = threshold
        self.recent_scores = deque(maxlen=smoothing_window)  # only stores last N values
        self.smoothed_avg = 0


    def estimate_image_quality(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            quality_score = self.quality_model.get_quality_score(img_tensor)
        quality_score = float(quality_score.item())
        self.recent_scores.append(quality_score)
        self.smoothed_avg = np.mean(self.recent_scores)

        return quality_score
    