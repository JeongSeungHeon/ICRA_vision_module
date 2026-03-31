import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a saved point-cloud .ply file.")
    parser.add_argument("ply_path", help="Path to the .ply point cloud file.")
    return parser.parse_args()


def main():
    args = parse_args()
    point_cloud = o3d.io.read_point_cloud(args.ply_path)
    print(point_cloud)
    o3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    main()
