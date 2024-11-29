import open3d as o3d
import numpy as np
import argparse

def pcd_to_bin(input_pcd, output_bin):
    # 加载 PCD 文件
    pcd = o3d.io.read_point_cloud(input_pcd)

    # 获取点云坐标和颜色数据
    points = np.asarray(pcd.points)  # shape (N, 3) for XYZ
    colors = np.asarray(pcd.colors)  # shape (N, 3) for RGB

    # 合并点云坐标和颜色，形成 [x, y, z, r, g, b] 格式
    # 如果颜色是浮点格式 [0, 1]，转换为 [0, 255] 整数
    if colors.max() > 1.0:
        print("Warning: colors are not in [0, 1] range, assuming [0, 255] range")
        print(colors.max())
    colors = (colors * 255).astype(np.uint8)  # 转为0-255范围的整数
    points_with_rgb = np.hstack((points, colors))

    # 保存为二进制 .bin 文件
    with open(output_bin, "wb") as f:
        f.write(points_with_rgb.astype(np.float32).tobytes())

    print(f"point cloud saved in '{output_bin}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PCD file with XYZRGB to a BIN file.")
    parser.add_argument("input_pcd", type=str, help="Path to the input PCD file.")
    parser.add_argument("output_bin", type=str, help="Path to the output BIN file.")
    
    args = parser.parse_args()
    pcd_to_bin(args.input_pcd, args.output_bin)
