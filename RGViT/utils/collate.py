import torch
from torch.nn.utils.rnn import pad_sequence

def radar_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch], dim=0)
    radar_seqs = [item['radar'] for item in batch]
    radar_padded = pad_sequence(radar_seqs, batch_first=True)  # (B, max_len, 6)

    radar_proj = [item['radar_projected'] for item in batch]
    radar_proj_padded = pad_sequence(radar_proj, batch_first=True)

    radar_labels = [torch.tensor(item['radar_labels'], dtype=torch.long) for item in batch]
    radar_labels_padded = pad_sequence(radar_labels, batch_first=True, padding_value=-1)

    gt_boxes_3d = [torch.tensor(item['gt_boxes_3d'], dtype=torch.float32) for item in batch]
    gt_labels = item['gt_labels'] if 'gt_labels' in item else None

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
