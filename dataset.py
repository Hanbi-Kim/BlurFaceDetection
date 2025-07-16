import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from config import CLASS_NAMES, label_map, IMG_SIZE, KERNEL_SIZE  # 외부 config 사용
from torchvision import transforms

class BlurredImageDataset(Dataset):
    def __init__(self, folder_path, use_edge=False, augment=False):
        self.image_paths = []
        self.labels = []
        self.use_edge = use_edge
        self.augment = augment

        for class_name in CLASS_NAMES:
            paths = glob(os.path.join(folder_path, class_name, '*.png'))
            self.image_paths.extend(paths)
            self.labels.extend([label_map[class_name]] * len(paths))

        self.original_len = len(self.image_paths)

        # 🔹 1배 증강 → 이미지 목록 2배로 복제
        if self.augment:
            self.image_paths *= 2
            self.labels *= 2

            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        blurred = cv2.blur(img, KERNEL_SIZE)

        if self.use_edge:
            gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
            edge = cv2.Laplacian(gray, cv2.CV_64F)
            edge = cv2.convertScaleAbs(edge)
            edge = cv2.resize(edge, IMG_SIZE)
            img = np.concatenate([blurred, edge[..., np.newaxis]], axis=2)  # (H, W, 4)
        else:
            img = blurred  # (H, W, 3)

        # 🔹 앞 절반은 원본, 뒷 절반은 증강
        if self.augment and idx >= self.original_len:
            img = self.augment_transform(img).numpy().transpose(1, 2, 0) * 255.0

        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)  # (C, H, W)

        return img, label

