import os
import torch
from config import get_config
from scripts.train import run_training
# from scripts.test import run_testing  # to be added later
# from scripts.visualize import run_visualization  # optional

def main():
    cfg = get_config()

    print("\n========== RGViT :: Radar-Guided Vision Transformer ==========")
    print(f"Mode       : {cfg.mode}")
    print(f"Device     : {cfg.device}")
    print(f"Batch Size : {cfg.batch_size}")
    print(f"Epochs     : {cfg.epochs}")
    print(f"LR         : {cfg.lr}\n")

    if cfg.mode == 'train':
        run_training(cfg)
    # elif cfg.mode == 'test':
    #     run_testing(cfg)
    # elif cfg.mode == 'visualize':
    #     run_visualization(cfg)
    else:
        raise NotImplementedError(f"Unknown mode: {cfg.mode}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
