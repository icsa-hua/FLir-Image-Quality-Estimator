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
    def __init__(self, quality_model_path='models/encoder_with_binary_head.pth'):
        encoder_model = IQAEncoder(feature_dim=128, model_name='resnet50').to(device)
        self.quality_model = DistortionBinaryClassifier(iqa_encoder=encoder_model).to(device)
        self.quality_model.load_state_dict(torch.load(quality_model_path, map_location=device))
        self.quality_model.eval()

    def estimate_image_quality(self, img):
        """Estimate raw image quality score."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            quality_score = self.quality_model.get_quality_score(img_tensor)
        return float(quality_score.item())
    
    def get_color(self, score):
        '''Get color based on quality score using gradient from 0.0 (red) to 1.0 (green).'''
        score = max(0.0, min(1.0, score))
        # Interpolate RGB values
        red = int((1 - score) * 255)
        green = int(score * 255)
        blue = 0
        
        return (blue, green, red)
        


class OnlineFLIQE(FLIQE):
    def __init__(self, quality_model_path='models/encoder_with_binary_head.pth', smoothing_window=300):
        super().__init__(quality_model_path)
        self.smoothing_window = smoothing_window
        self.sessions = {}

    def create_session(self, session_id):
        self.sessions[session_id] = {
            'recent_scores': deque(maxlen=self.smoothing_window),
            'smoothed_avg': 0
        }

    def estimate_smoothed_quality(self, img: np.ndarray, session_id: str):
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found.")
        
        quality_score = self.estimate_image_quality(img)
        session_data = self.sessions[session_id]
        session_data['recent_scores'].append(quality_score)
        session_data['smoothed_avg'] = np.mean(session_data['recent_scores'])
        return session_data['smoothed_avg']

    def get_smoothed_quality(self, session_id):
        if session_id not in self.sessions:
            return 0
        return self.sessions[session_id]['smoothed_avg']

    def remove_session(self, session_id):
        self.sessions.pop(session_id, None)

    