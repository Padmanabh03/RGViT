import matplotlib.pyplot as plt
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import numpy as np

def render_3d_boxes(nusc, sample, cam_calib, cam_intrinsic, img_np):
    plt.figure(figsize=(10, 5))
    plt.imshow(img_np)

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        box = nusc.get_box(ann_token)
        box.translate(-np.array(cam_calib['translation']))
        box.rotate(Quaternion(cam_calib['rotation']).inverse)
        corners_2d = view_points(box.corners(), cam_intrinsic, normalize=True)

        for i in range(4):
            x = [corners_2d[0, i], corners_2d[0, i + 4]]
            y = [corners_2d[1, i], corners_2d[1, i + 4]]
            plt.plot(x, y, 'b-')

        front = [0, 1, 2, 3, 0]
        back = [4, 5, 6, 7, 4]
        plt.plot(corners_2d[0, front], corners_2d[1, front], 'b-')
        plt.plot(corners_2d[0, back], corners_2d[1, back], 'b-')

    plt.title('3D Bounding Boxes on CAM_FRONT')
    plt.axis('off')
    plt.show()