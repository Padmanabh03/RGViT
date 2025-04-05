import numpy as np
from pyquaternion import Quaternion

def get_transform(translation, rotation):
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T

def transform_pointcloud(pc, source_sd, target_sd, nusc):
    source_cs = nusc.get('calibrated_sensor', source_sd['calibrated_sensor_token'])
    source_ep = nusc.get('ego_pose', source_sd['ego_pose_token'])
    target_cs = nusc.get('calibrated_sensor', target_sd['calibrated_sensor_token'])
    target_ep = nusc.get('ego_pose', target_sd['ego_pose_token'])

    T_source_to_ego = get_transform(source_cs['translation'], source_cs['rotation'])
    T_ego_to_global = get_transform(source_ep['translation'], source_ep['rotation'])
    T_global_to_target_ego = np.linalg.inv(get_transform(target_ep['translation'], target_ep['rotation']))
    T_target_ego_to_sensor = np.linalg.inv(get_transform(target_cs['translation'], target_cs['rotation']))

    full_T = T_target_ego_to_sensor @ T_global_to_target_ego @ T_ego_to_global @ T_source_to_ego
    pc.transform(full_T)
    return pc

def transform_to_global(pc, sample_data, nusc):
    cs = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sample_data['ego_pose_token'])
    T = get_transform(ep['translation'], ep['rotation']) @ get_transform(cs['translation'], cs['rotation'])
    pc.transform(T)
    return pc
