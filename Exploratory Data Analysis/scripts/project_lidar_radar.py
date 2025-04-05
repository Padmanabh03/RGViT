import matplotlib.pyplot as plt

def project_points_on_image(img_np, lidar_pts, radar_pts):
    plt.figure(figsize=(12, 8))
    plt.imshow(img_np)
    plt.scatter(lidar_pts[0], lidar_pts[1], s=2, c='green', label='LiDAR')
    plt.scatter(radar_pts[0], radar_pts[1], s=2, c='red', label='Radar')
    plt.title('Projected LiDAR & Radar Points on CAM_FRONT')
    plt.axis('off')
    plt.legend()
    plt.show()

