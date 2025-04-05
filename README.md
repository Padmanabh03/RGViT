# Radar-Guided Vision Transformer with Cross-Attention Fusion for Object Detection and Depth Estimation

Author: Padmanabh Butala
Email: pb8176@rit.edu
Institution: Rochester Institute of Technology



## Project Overview

This repository contains Python scripts for exploratory data analysis (EDA) on a **mini subset of the nuScenes dataset**, designed to visualize and understand radar, LiDAR, and camera sensor data. It serves as a foundation for the full capstone project: **Radar-Guided Vision Transformer with Cross-Attention Fusion for Object Detection and Depth Estimation in Autonomous Vehicles**.

---

## Directory Structure

```
RadarGuidedViT/
│
├── scripts/                     # All reusable Python modules
│   ├── config.py               # Configs for data path and version
│   ├── load_nuscenes.py       # Initializes the NuScenes dataset
│   ├── extract_data.py        # Extracts the .tgz file
│   ├── visualize_camera_image.py
│   ├── project_lidar_radar.py
│   ├── render_3d_boxes.py
│   ├── visualize_open3d.py
│   ├── export_pointcloud.py
│   └── helpers.py             # Common utilities for transformation
│
├── README.md                   # You're here!
├── requirements.txt            # Python dependencies
└── run_eda_demo.py             # Sample script to run the whole EDA pipeline
```

---

## Dataset Information

We use a **subset of the [nuScenes](https://www.nuscenes.org/nuscenes#)** dataset:

- **Mini Split:** 10 scenes (for fast experimentation)
- **File size:** ~4 GB (downloadable [here](https://www.nuscenes.org/download))
- **Full Dataset:** ~300 GB (not needed for this repo)
- **Sensors:** 6 surround-view cameras, 5 radars, and 1 LiDAR
- Note: The mini set still contains synchronized, annotated multi-modal data suitable for EDA.

---

## Setup Instructions

### 1. Recommended Python Version
We recommend **Python 3.10.x** due to compatibility with:
- `nuscenes-devkit`
- `open3d`
- `pyquaternion`

### 2. Environment Setup

```bash
# Clone the repo
git clone https://github.com/your_username/RadarGuidedViT.git
cd RadarGuidedViT

# (Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install required packages
pip install -r requirements.txt
```

Sample `requirements.txt`:
```
nuscenes-devkit
pyquaternion
matplotlib
numpy
Pillow
open3d
```

---

## Data Preparation

### Step 1: Download the mini dataset from [nuScenes Download Page](https://www.nuscenes.org/download)  
Download the `v1.0-mini.tgz` (~4 GB).

### Step 2: Extract the `.tgz` file
```bash
python scripts/extract_data.py
```

This will extract files to:
```
C:\Users\padma\OneDrive\Desktop\Python\RadarGuidedViT\Exploratory Data Analysis\nuscenes
```

Make sure to update `scripts/config.py` if your data directory differs.

---


The pipeline includes:
- Loading front camera images
- Projecting LiDAR/Radar onto image plane
- Rendering 3D bounding boxes
- Visualizing point clouds with Open3D

---

##  Key Results & Visualizations

| Visualization Type                     | Output Example |
|----------------------------------------|----------------|
| Front camera image                     | ![CAM_FRONT](docs/cam_front.jpg) |
| LiDAR & Radar projection (image space) | ![Lidar+Radar](docs/lidar_radar_overlay.jpg) |
| 3D Bounding Boxes                      | ![3D Boxes](docs/3d_boxes.jpg) |
| Open3D 3D Scatter                      | ![Open3D](docs/open3d_scene.jpg) |

---

## References & Acknowledgements

- Code uses the official [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
- Dataset: nuScenes by [Motional](https://www.nuscenes.org/)
- Transformer-based fusion methodology inspired by [CRAFT](https://arxiv.org/abs/2303.12250), HVDetFusion, and ClusterFusion.





