import cv2
from torch.utils.data import Dataset
from distortions import *


class ImageDataset(Dataset):
    def __init__(self, image_paths, distortions=None, transform=None, is_cache_images=False, binary_labels=False):
        self.image_paths = image_paths
        self.distortions = distortions
        self.transform = transform
        self.images = [cv2.imread(p) for p in image_paths] if is_cache_images else None
        self.binary_labels = binary_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.images[idx] if self.images else cv2.imread(self.image_paths[idx])

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply distortion 
        if self.binary_labels:
            if self.distortions and random.random() < 0.5:
                img, label = RandomDistortion(self.distortions)(img)
            else:
                label = "Clean"
        else:
            if self.distortions:
                img, label = RandomDistortion(self.distortions)(img)
            else:
                label = "Clean"

        # Transform image
        if self.transform:
            img = self.transform(img)

        return img, label


class VideoFrameDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform

        # Get total number of frames
        cap = cv2.VideoCapture(video_path)
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Frame {idx} not found in {self.video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

        if self.transform:
            frame, distortion = self.transform(frame)
        else:
            distortion = "None"

        return frame, distortion


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader    

    transform = transforms.Compose([
        lambda img: RandomDistortion([
            LensBlur(ksize=7),
            MotionBlur(degree=10, angle=30),
            Blackout()
        ], p=0.7)(img),  # returns (img, label)
        lambda tup: (transforms.ToPILImage()(tup[0]), tup[1]),
        lambda tup: (transforms.Resize((224, 224))(tup[0]), tup[1]),
        lambda tup: (transforms.ToTensor()(tup[0]), tup[1]),
    ])

    dataset = VideoFrameDataset("data/1.mp4", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for batch in dataloader:
        imgs = batch[0]
        labels = batch[1]
        print(f"Batch shape: {imgs.shape}, unique labels: {set(labels)}, total classes: {len(set(labels))}")
        cv2.imshow("Frame", imgs[0].permute(1, 2, 0).numpy())
        cv2.waitKey(0)