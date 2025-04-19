import sys
import os
from multiprocessing import freeze_support

# Make sure imports work when spawning workers on Windows
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import get_config
from models.rgvit_net import RGVisionTransformer
from data.nuscenes_dataset import NuScenesRadarCameraDataset, radar_collate_fn
from utils.metrics import compute_3d_detection_metrics


def run_training():
    cfg = get_config()
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # -------------------- DATASET --------------------
    dataset = NuScenesRadarCameraDataset(
        json_path=cfg.json_path,
        blobs_root=cfg.blobs_root
    )
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=radar_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=radar_collate_fn,
        pin_memory=True
    )

    # -------------------- MODEL ----------------------
    model = RGVisionTransformer(num_classes=cfg.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_reg = nn.SmoothL1Loss()

    # -------------------- TRAINING -------------------
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = cls_loss_total = reg_loss_total = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
        for batch in loop:
            images = batch['image'].to(device)
            radar = batch['radar'].to(device)
            labels = batch['labels'].to(device)
            gt_boxes_3d = batch['gt_boxes_3d']

            optimizer.zero_grad()
            cls_logits, box_preds = model(images, radar)

            # Classification: mode label, ignore -1
            valid = labels.clone().masked_fill(labels == -1, 9999)
            mode_labels = torch.mode(valid, dim=1).values
            mode_labels = mode_labels.masked_fill(mode_labels == 9999, 0).long()
            cls_loss = criterion_cls(cls_logits, mode_labels)

            # Regression target from first 3D box if available
            reg_targets = []
            for boxes in gt_boxes_3d:
                if boxes.shape[0] > 0:
                    reg_targets.append(boxes[0][:4].to(device))  # x,y,w,l
                else:
                    reg_targets.append(torch.zeros(4, device=device))
            reg_targets = torch.stack(reg_targets)
            reg_loss = criterion_reg(box_preds, reg_targets)

            loss = cls_loss + reg_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            cls_loss_total += cls_loss.item()
            reg_loss_total += reg_loss.item()
            loop.set_postfix(loss=loss.item(), cls=cls_loss.item(), reg=reg_loss.item())

        # ------------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        preds_by_sample = {}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]"):
                images = batch['image'].to(device)
                radar = batch['radar'].to(device)
                labels = batch['labels'].to(device)
                gt_boxes_3d = batch['gt_boxes_3d']
                sample_tokens = batch['sample_token']

                cls_logits, box_preds = model(images, radar)

                # Classification loss
                valid = labels.clone().masked_fill(labels == -1, 9999)
                mode_labels = torch.mode(valid, dim=1).values
                mode_labels = mode_labels.masked_fill(mode_labels == 9999, 0).long()
                cls_loss = criterion_cls(cls_logits, mode_labels)

                # Regression loss
                reg_targets = []
                for boxes in gt_boxes_3d:
                    if boxes.shape[0] > 0:
                        reg_targets.append(boxes[0][:4].to(device))
                    else:
                        reg_targets.append(torch.zeros(4, device=device))
                reg_targets = torch.stack(reg_targets)
                reg_loss = criterion_reg(box_preds, reg_targets)

                val_loss += (cls_loss + reg_loss).item()

                # Build predictions dict for nuScenes eval
                for i, token in enumerate(sample_tokens):
                    if gt_boxes_3d[i].shape[0] > 0:
                        # center (x,y,z)
                        center = box_preds[i][:3].cpu().tolist()
                        # placeholder dims (w,l,h)
                        dims = [0.0, 0.0, 0.0]
                        # identity quaternion
                        rot = [0.0, 0.0, 0.0, 1.0]
                        score = torch.softmax(cls_logits[i], dim=0).max().item()
                        class_id = int(torch.argmax(cls_logits[i]).item())

                        # [x,y,z, w,l,h, [qx,qy,qz,qw], class_id, score]
                        preds_by_sample[token] = [[
                            *center,
                            *dims,
                            rot,
                            class_id,
                            score
                        ]]

        # --- logging & checkpointing ---
        print(f"\n[Epoch {epoch+1}] "
              f"Train Loss: {train_loss:.4f} | Cls: {cls_loss_total:.4f} | Reg: {reg_loss_total:.4f}")
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}")

        ckpt_path = os.path.join(cfg.save_dir, f"rgvit_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Model saved to {ckpt_path}")

        # --- nuScenes evaluation ---
        print("\n📊 Running NuScenes-style Evaluation...")
        metrics = compute_3d_detection_metrics(
            preds_by_sample,
            nusc_root=cfg.nusc_root,
            version=cfg.version,
            eval_set="val"
        )
        print(f"📈  mAP: {metrics['mean_ap']:.4f}, NDS: {metrics['nd_score']:.4f}")

    print("🎉 Training complete!")

if __name__ == '__main__':
    freeze_support()
    run_training()
