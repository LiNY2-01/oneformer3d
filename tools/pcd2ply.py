import open3d as o3d
import numpy as np
import argparse

def pcd_to_bin(input_pcd, output_ply):
    # 加载 PCD 文件
    pcd = o3d.io.read_point_cloud(input_pcd)

    # 保存为二进制 .ply 文件
    o3d.io.write_point_cloud(output_ply, pcd)

    print(f"point cloud saved in '{output_ply}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PCD file with XYZRGB to a BIN file.")
    parser.add_argument("input_pcd", type=str, help="Path to the input PCD file.")
    parser.add_argument("output_ply", type=str, help="Path to the output ply file.")
    
    args = parser.parse_args()
    pcd_to_bin(args.input_pcd, args.output_ply)
