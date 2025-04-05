import open3d as o3d

def visualize_open3d(lidar_pc, radar_pc, boxes):
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    geometries = [lidar_pc, radar_pc, coord_frame] + boxes
    o3d.visualization.draw_geometries(geometries)