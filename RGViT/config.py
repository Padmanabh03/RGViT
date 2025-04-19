import argparse

def get_config():
    parser = argparse.ArgumentParser(description="RGViT Training Config")

    # ─── Data paths ───────────────────────────────────────────────────────────
    parser.add_argument(
        '--json_path',
        type=str,
        default=r"C:/Users/padma/OneDrive/Desktop/Python/RadarGuidedViT/nuscenes_blobs/train_samples.json",
        help="Path to the train_samples.json you generated."
    )
    parser.add_argument(
        '--blobs_root',
        type=str,
        default=r"C:/Users/padma/OneDrive/Desktop/Python/RadarGuidedViT/nuscenes_blobs",
        help="Root of your nuscenes_blobs directory (images + radar files)."
    )
    parser.add_argument(
        '--nusc_root',
        type=str,
        default=r"C:/Users/padma/OneDrive/Desktop/Python/RadarGuidedViT/nuscenes_meta",
        help="Parent folder containing your metadata subfolders."
    )
    parser.add_argument(
        '--version',
        type=str,
        default="subset_trainval01",
        help="Which metadata subfolder to use under nusc_root (e.g. 'subset_trainval01' or 'v1.0-trainval')."
    )

    # ─── Model & training ────────────────────────────────────────────────────
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help="Batch size for training / validation."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="Number of training epochs."
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help="Learning rate."
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=11,
        help="Number of object classes (including background if used)."
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help="Number of DataLoader worker processes."
    )

    # ─── Runtime mode ─────────────────────────────────────────────────────────
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'visualize'],
        default='train',
        help="Which pipeline to run."
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help="Device for torch (e.g. 'cpu' or 'cuda')."
    )

    return parser.parse_args()


