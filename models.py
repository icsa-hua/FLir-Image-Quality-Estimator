import torch
# print("CUDA version: ", torch.version.cuda)
# print("Available CUDA architectures: ", torch.cuda.get_arch_list())
import torch.nn as nn
import torchvision.models as models
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


models_dict = {
    'resnet18': {'model': models.resnet18, 'output_dim': 512, 'pre_trained_weights': models.ResNet18_Weights.IMAGENET1K_V1},
    'resnet50': {'model': models.resnet50, 'output_dim': 2048, 'pre_trained_weights': models.ResNet50_Weights.IMAGENET1K_V1},
    'vit_b_16': {'model': models.vit_b_16, 'output_dim': 768, 'pre_trained_weights': models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1}
}


class IQAEncoder(nn.Module):
    def __init__(self, feature_dim=128, model_name='resnet18', pretrained=True, freeze_encoder=True):
        super().__init__()
        
        backbone = models_dict[model_name]['model']
        self.encoder_output_dim = models_dict[model_name]['output_dim']
        encoder_weights = models_dict[model_name]['pre_trained_weights'] if pretrained else None

        # Initialize encoder
        encoder = backbone(weights=encoder_weights)
        if model_name.startswith('vit'):
            encoder.heads = nn.Identity()
        else:
            encoder.fc = nn.Identity()
        self.encoder = encoder

        # Optionally freeze encoder parameters
        if pretrained and freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection head
        # Dynamically create projection layers with halving dimensions
        dims = []
        dim = self.encoder_output_dim
        while dim > feature_dim:
            next_dim = max(dim // 2, feature_dim)  # Prevent going below target
            dims.append((dim, next_dim))
            if next_dim == feature_dim:
                break
            dim = next_dim

        layers = []
        for i, (in_dim, out_dim) in enumerate(dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 1:  # Apply BatchNorm and ReLU only between layers
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))

        self.projection_head = nn.Sequential(*layers)

    def forward(self, x):
        features = self.encoder(x)
        out = self.projection_head(features)
        return nn.functional.normalize(out, dim=1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(features.shape[0], device=device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss
    

class DistortionBinaryClassifier(nn.Module):
    def __init__(self, iqa_encoder: IQAEncoder, hidden_dims=(128, 64), dropout_p=0.3):
        super().__init__()
        self.iqa_encoder = iqa_encoder        
        for param in self.iqa_encoder.parameters():
            param.requires_grad = False  # Freeze encoder
        self.iqa_encoder.eval()

        input_dim = self.iqa_encoder.projection_head[-1].out_features

        layers = []
        last_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(last_dim, hdim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(hdim))
            # layers.append(nn.Dropout(p=dropout_p))  # <-- drop out for regularization
            last_dim = hdim

        # Final layer: output logits for binary classification
        layers.append(nn.Linear(last_dim, 1))  # Output is a single logit
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():  # Prevent gradient updates to encoder
            features = self.iqa_encoder(x)
        return self.classifier(features)
    
    def get_distortion_score(self, x):
        """Get distortion score (higher = more distorted)"""
        logits = self.forward(x)
        return torch.sigmoid(logits)  # 0-1 range, higher = more distorted
    
    def get_quality_score(self, x):
        """Get quality score (higher = better quality)"""
        distortion_prob = self.get_distortion_score(x)
        return 1 - distortion_prob  # Invert: higher = better quality
    
    def get_raw_logits(self, x):
        """Get raw logits for advanced use cases"""
        return self.forward(x)
    

def extract_features(model, dataloader, label_map, device):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        try:
            batch_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting features", leave=False)
            for batch_idx, (imgs, labels) in batch_bar:
                imgs = imgs.to(device)
                labels = torch.tensor([label_map[l] for l in labels], dtype=torch.long)
                features = model(imgs).cpu()
                all_features.append(features)
                all_labels.append(labels)
        except ValueError as e:
            print(f"Error during feature extraction: {e}")
    return torch.cat(all_features), torch.cat(all_labels)


def plot_tsne(features, labels, label_map, epoch, model_name, dim_out, avg_loss):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(features.numpy())

    plt.figure(figsize=(8, 6))
    num_classes = len(label_map)
    for i in range(num_classes):
        idx = labels.numpy() == i
        label_name = list(label_map.keys())[list(label_map.values()).index(i)]
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label_name, alpha=0.6, s=10)

    plt.legend()
    plt.title(f"t-SNE of Embeddings (Epoch {epoch} - {model_name}, Dim: {dim_out}, Loss: {avg_loss:.4f})")
    path = f"eval_results/{model_name}_{dim_out}_out"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/tsne_epoch_{epoch}.png")
    plt.close()


if __name__ == "__main__":
    from datasets import ImageDataset
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from distortions import *

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("CUDA version: ", torch.version.cuda)
    model_name = 'resnet50'  # 'resnet18' or 'resnet50', 'vit_b_16'
    dim_out = 128
    model = IQAEncoder(feature_dim=dim_out, model_name=model_name).to(device)
    print("Model architecture: ", model.projection_head)
    criterion = SupConLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    distortions = [Clean(), LensBlur(), MotionBlur(), GaussianNoise(), Overexposure(), Underexposure(), Compression(), Ghosting(), Aliasing()]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), # 224 or 384
        transforms.ToTensor(),
    ])

    image_folder = "data/FLIR_ADAS_v2/images_thermal_train/data"
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.jpg', '.png'))]
    dataset = ImageDataset(image_paths, distortions=distortions, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    print(f"Train Dataset length: {len(dataset)}")

    image_folder = "data/FLIR_ADAS_v2/images_thermal_val/data"
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.jpg', '.png'))]
    eval_dataset = ImageDataset(image_paths, distortions=distortions, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)
    print(f"Eval Dataset length: {len(eval_dataset)}")

    label_map = {distortion.__class__.__name__: i for i, distortion in enumerate(distortions)}

    epochs = 100
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = len(dataloader)
        batch_bar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch + 1}", leave=False)
        for batch_idx, (imgs, labels) in batch_bar:
            imgs = imgs.to(device)
            labels = torch.tensor([label_map[l] for l in labels], dtype=torch.long, device=device)
            features = model(imgs)
            loss = criterion(features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / total_batches
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

        # Validation and early stopping
        if ((epoch+1) % 10 == 0) or (epoch == 0):
            # Validation loss for early stopping
            model.eval()
            eval_loss = 0.0
            all_features = []
            all_labels = []
            with torch.no_grad():
                batch_bar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Extracting features for validation", leave=False)
                for batch_idx, (imgs, labels) in batch_bar:
                    imgs = imgs.to(device)
                    labels = torch.tensor([label_map[l] for l in labels], dtype=torch.long)
                    features = model(imgs).cpu()
                    all_features.append(features)
                    all_labels.append(labels)
                    loss = criterion(features, labels)
                    eval_loss += loss.item()
            eval_loss /= len(eval_dataloader)
            eval_features, eval_labels = torch.cat(all_features), torch.cat(all_labels)
            print(f"Validation Loss: {eval_loss:.4f}")
            plot_tsne(eval_features, eval_labels, label_map, epoch+1, model_name, dim_out, eval_loss)
            if eval_loss + 0.05 < best_loss:
                best_loss = eval_loss
            else:
                print(f"Early stopping at epoch {epoch + 1} with loss {eval_loss:.4f} (best: {best_loss:.4f})")
                break
        

    # Save model checkpoint
    model_path = f"models/{model_name}_{dim_out}_out.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # test the model with tsne at data/video_frames_1
    image_folder = "data/video_frames_1"
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.jpg', '.png'))]
    test_dataset = ImageDataset(image_paths, distortions=distortions, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    print(f"Test Dataset length: {len(test_dataset)}")

    test_features, test_labels = extract_features(model, test_dataloader, label_map, device)
    plot_tsne(test_features, test_labels, label_map, 'TEST', model_name, dim_out, best_loss)
