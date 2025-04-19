import os
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import numpy as np
import torchvision.transforms as T

CLASS_MAPPING = {
    'car': 1,
    'truck': 2,
    'bus': 3,
    'trailer': 4,
    'construction_vehicle': 5,
    'pedestrian': 6,
    'motorcycle': 7,
    'bicycle': 8,
    'traffic_cone': 9,
    'barrier': 10
}

class NuScenesRadarCameraDataset(Dataset):
    def __init__(self, json_path, blobs_root, transform=None):
        self.json_path = json_path
        self.blobs_root = blobs_root
        self.transform = transform if transform is not None else self.default_transform()

        with open(self.json_path, 'r') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = os.path.join(self.blobs_root, sample['image_path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        radar_pts_cam = torch.tensor(sample['radar_points_cam'], dtype=torch.float32)  # (N, 4)
        radar_feats = torch.tensor(sample['radar_features'], dtype=torch.float32)      # (N, 3)
        radar_data = torch.cat([radar_pts_cam[:, :3], radar_feats], dim=1)              # (N, 6): x,y,z,vx,vy,rcs

        radar_projected = torch.tensor(sample['radar_projected'], dtype=torch.float32)  # (N, 2)

        radar_labels = torch.tensor(sample.get('radar_labels', [-1] * len(sample['radar_projected'])), dtype=torch.long)

        gt_boxes_3d = torch.tensor(sample['gt_boxes_3d'], dtype=torch.float32) if sample.get('gt_boxes_3d') else torch.zeros((0, 7))
        gt_labels_str = sample.get('gt_labels', [])
        gt_labels = torch.tensor([CLASS_MAPPING.get(lbl, 0) for lbl in gt_labels_str], dtype=torch.long)

        return {
            'image': image,
            'radar': radar_data,
            'radar_projected': radar_projected,
            'labels': radar_labels,
            'gt_boxes_3d': gt_boxes_3d,
            'gt_labels': gt_labels,
            'sample_token': sample['sample_token'],
            'camera': sample['camera'],
            'radar_sensor': sample.get('radar', 'RADAR_FRONT')
        }

    def default_transform(self):
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])

def radar_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch], dim=0)
    radar_seqs = [item['radar'] for item in batch]
    radar_padded = pad_sequence(radar_seqs, batch_first=True)

    radar_proj = [item['radar_projected'] for item in batch]
    radar_proj_padded = pad_sequence(radar_proj, batch_first=True)

    radar_labels = [item['labels'] for item in batch]
    radar_labels_padded = pad_sequence(radar_labels, batch_first=True, padding_value=-1)

    gt_boxes_3d = [item['gt_boxes_3d'] for item in batch]
    gt_labels = [item['gt_labels'] for item in batch]

    return {
        'image': images,
        'radar': radar_padded,
        'radar_projected': radar_proj_padded,
        'labels': radar_labels_padded,
        'gt_boxes_3d': gt_boxes_3d,
        'gt_labels': gt_labels,
        'sample_token': [item['sample_token'] for item in batch],
        'camera': [item['camera'] for item in batch],
        'radar_sensor': [item['radar_sensor'] for item in batch]
    }

if __name__ == "__main__":
    json_path = "C:/Users/padma/OneDrive/Desktop/Python/RadarGuidedViT/nuscenes_blobs/train_samples.json"
    blobs_root = "C:/Users/padma/OneDrive/Desktop/Python/RadarGuidedViT/nuscenes_blobs"

    dataset = NuScenesRadarCameraDataset(json_path=json_path, blobs_root=blobs_root)
    print(f"Loaded {len(dataset)} samples")
    sample = dataset[0]
    for k, v in sample.items():
        print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
