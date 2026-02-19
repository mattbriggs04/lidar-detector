import numpy as np
import glob

def load_velodyne_bin(bin_path: str) -> np.ndarray:
    pc = np.fromfile(bin_path, dtype=np.float32)
    pc = pc.reshape(-1, 4)
    return pc

if __name__ == "__main__":
    train_velo_path = "data/training/velodyne"
    lidar_files = sorted(glob.glob(train_velo_path + "/*.bin"))

    print(f"Found {len(lidar_files)} training LiDAR files")
    pc = load_velodyne_bin(lidar_files[0])

    print("Pointcloud shape:", pc.shape)
    print("x range:", pc[:,0].min(), pc[:,0].max())
    print("y range:", pc[:,1].min(), pc[:,1].max())
    print("z range:", pc[:,2].min(), pc[:,2].max())
    print("reflectance range:", pc[:,3].min(), pc[:,3].max())
