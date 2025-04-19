import os
import torch
import cv2
import numpy as np
from data.nuscenes_dataset import NuScenesRadarCameraDataset
from config import get_config
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def draw_radar_on_image(image, radar_points, labels=None):
    img_np = np.array(image)
    for i, pt in enumerate(radar_points):
        u, v = int(pt[0]), int(pt[1])
        color = (0, 255, 0)  # Green default
        if labels is not None and labels[i] != -1:
            color = (255, 0, 0)  # Blue if labeled
        cv2.circle(img_np, (u, v), radius=3, color=color, thickness=-1)
    return img_np

def run_visualization(cfg):
    dataset = NuScenesRadarCameraDataset(cfg.json_path, cfg.blobs_root)
    print(f"✅ Loaded {len(dataset)} samples from: {cfg.json_path}\n")

    for i in range(10):
        sample = dataset[i]
        img = sample['image']
        radar_uv = sample['radar_projected'].numpy()
        labels = sample['labels'].numpy()

        overlay = draw_radar_on_image(to_pil_image(img.cpu()), radar_uv, labels)
        plt.figure(figsize=(12, 6))
        plt.imshow(overlay)
        plt.title(f"Sample {i} - CAM: {sample['camera']}, RADAR: {sample['radar_sensor']}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        input("Press Enter to continue...")