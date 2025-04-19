import os
import json
import torch
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix, BoxVisibility

# Define the paths to the metadata and blobs (calibration, images, point clouds, etc.)
META_ROOT = './nuscenes_meta'
BLOBS_ROOT = './nuscenes_blobs'
SAVE_PATH = os.path.join(BLOBS_ROOT, 'train_samples.json')

# Define the camera-radar sensor pairs (as used in CenterFusion)
CAMERA_RADAR_PAIRS = [
    ('CAM_FRONT', 'RADAR_FRONT'),
    ('CAM_FRONT_LEFT', 'RADAR_FRONT_LEFT'),
    ('CAM_FRONT_RIGHT', 'RADAR_FRONT_RIGHT'),
    ('CAM_BACK_LEFT', 'RADAR_BACK_LEFT'),
    ('CAM_BACK_RIGHT', 'RADAR_BACK_RIGHT')
]

# Mapping for object classes (CenterFusion usually uses similar mappings)
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

def load_transform(nusc, sample_data_token):
    """
    Load the transformation matrices for a given sample_data token.
    
    This function retrieves the sensor-to-ego and ego-to-global transformation
    matrices, which are then used to project points between coordinate systems.
    """
    sd = nusc.get('sample_data', sample_data_token)
    sensor = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
    sensor_from_ego = transform_matrix(sensor['translation'], Quaternion(sensor['rotation']), inverse=False)
    ego_from_global = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
    return sensor_from_ego, ego_from_global, sensor

def project_radar_to_image(nusc, radar_token, cam_token):
    """
    Projects radar points into the image plane.
    
    This function loads the radar point cloud (and selects the first three coordinates),
    applies the necessary sensor-to-ego and ego-to-camera transforms, filters out points
    that fall behind the camera, and then projects the remaining points into the image using
    the intrinsic matrix of the camera.
    """
    radar_data = nusc.get('sample_data', radar_token)
    radar_path_pcd = radar_data['filename']
    radar_abs = os.path.join(BLOBS_ROOT, radar_path_pcd)

    # Load the radar point cloud; note that indices 8, 9 and 6 are velocity x, velocity y and RCS
    radar_pc = RadarPointCloud.from_file(radar_abs)
    radar_points = radar_pc.points[:3, :]  # x, y, z coordinates
    radar_vx = radar_pc.points[8, :]
    radar_vy = radar_pc.points[9, :]
    radar_rcs = radar_pc.points[6, :]

    # Get the transformation for both radar and camera sensors
    radar_sensor_from_ego, _, _ = load_transform(nusc, radar_token)
    camera_sensor_from_ego, _, cam_sensor = load_transform(nusc, cam_token)

    # Transform radar points: sensor -> ego -> camera coordinates
    radar_in_ego = np.dot(radar_sensor_from_ego, np.vstack((radar_points, np.ones((1, radar_points.shape[1])))))
    radar_in_cam = np.dot(np.linalg.inv(camera_sensor_from_ego), radar_in_ego)

    # Filter out radar points behind the camera (z <= 0)
    front_mask = radar_in_cam[2, :] > 0
    radar_in_cam = radar_in_cam[:, front_mask]
    radar_vx = radar_vx[front_mask]
    radar_vy = radar_vy[front_mask]
    radar_rcs = radar_rcs[front_mask]

    # Project radar points to the image plane
    proj_matrix = np.array(cam_sensor['camera_intrinsic'])
    image_points = np.dot(proj_matrix, radar_in_cam[:3, :])
    # Normalize homogeneous coordinates
    image_points = image_points[:2, :] / image_points[2, :]

    radar_projected = image_points.T
    radar_points_cam = radar_in_cam.T
    radar_features = np.stack([radar_vx, radar_vy, radar_rcs], axis=1)

    return radar_projected, radar_points_cam, radar_features

def get_3d_boxes(nusc, cam_token):
    """
    Retrieves 3D bounding boxes and their corresponding labels from a camera sample.
    
    Note: We use a corrected extraction of the object class by using the last token 
    in the object name (i.e. "vehicle.car" -> "car") which is consistent with CLASS_MAPPING.
    """
    _, boxes, _ = nusc.get_sample_data(
        cam_token,
        box_vis_level=BoxVisibility.ANY
    )
    gt_boxes = []
    gt_labels = []
    for box in boxes:
        # Correct extraction: use the last token of the object name
        class_name = box.name.split('.')[-1]
        if class_name not in CLASS_MAPPING:
            continue
        gt_boxes.append([
            box.center[0], box.center[1], box.center[2],
            box.wlh[0], box.wlh[1], box.wlh[2],
            box.orientation.yaw_pitch_roll[0]  # Only the yaw is used in many detection systems
        ])
        gt_labels.append(class_name)
    return gt_boxes, gt_labels

def extract_dataset():
    """
    Extracts the dataset samples by iterating through all nuScenes samples and fusing 
    radar with camera data. This function builds a list of paired samples similar to
    what CenterFusion uses as input.
    
    It performs the following:
      1. Loads the nuScenes data.
      2. Iterates over every sample in the dataset.
      3. For each camera-radar sensor pair, checks if both image and radar files exist.
      4. Projects radar points into the image frame.
      5. Retrieves ground truth 3D boxes and labels.
      6. Saves the paired sample if both radar points and GT boxes are nonempty.
    """
    nusc = NuScenes(version='v1.0-trainval', dataroot=META_ROOT, verbose=True)
    paired_samples = []

    for sample in tqdm(nusc.sample, desc="Processing train samples"):
        for cam, radar in CAMERA_RADAR_PAIRS:
            try:
                cam_token = sample['data'][cam]
                radar_token = sample['data'][radar]
                cam_data = nusc.get('sample_data', cam_token)
                radar_data = nusc.get('sample_data', radar_token)

                image_path = cam_data['filename']
                radar_path_pcd = radar_data['filename']
                # The code assumes that you have pre-converted radar point clouds to .pt format
                radar_path_pt = radar_path_pcd.replace('.pcd', '.pt')

                image_abs = os.path.join(BLOBS_ROOT, image_path)
                radar_abs_check = os.path.join(BLOBS_ROOT, radar_path_pcd)

                # Skip this sample if the image or radar file is missing
                if not (os.path.exists(image_abs) and os.path.exists(radar_abs_check)):
                    continue

                # Project radar into the image plane and extract radar features
                radar_projected, radar_points_cam, radar_features = project_radar_to_image(nusc, radar_token, cam_token)
                # Get corresponding ground truth 3D boxes and labels
                gt_boxes_3d, gt_labels = get_3d_boxes(nusc, cam_token)

                # Optionally, you can skip samples where no valid radar or GT exists
                if len(radar_projected) == 0 or len(gt_boxes_3d) == 0:
                    continue

                paired_samples.append({
                    'image_path': image_path,
                    'radar_path': radar_path_pt,
                    'radar_projected': radar_projected.tolist(),
                    'radar_points_cam': radar_points_cam.tolist(),
                    'radar_features': radar_features.tolist(),
                    'gt_boxes_3d': gt_boxes_3d,
                    'gt_labels': gt_labels,
                    'sample_token': sample['token'],
                    'camera': cam,
                    'radar_sensor': radar
                })
            except Exception as e:
                print(f"Skipping {cam} + {radar} due to error: {e}")
                continue

    print(f"\n✅ Total paired samples: {len(paired_samples)}")
    with open(SAVE_PATH, 'w') as f:
        json.dump(paired_samples, f, indent=2)
    print(f"📝 Saved to {SAVE_PATH}")

if __name__ == '__main__':
    extract_dataset()
