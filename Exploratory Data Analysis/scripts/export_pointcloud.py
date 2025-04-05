def save_point_cloud_to_ply(filename, points, colors):
    num_points = points.shape[1]
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(num_points):
            x, y, z = points[0, i], points[1, i], points[2, i]
            r, g, b = colors[i]
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")
    print(f"Saved combined point cloud to {filename}")