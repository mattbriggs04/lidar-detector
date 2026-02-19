import numpy as np
import glob

def load_velodyne_bin(bin_path: str) -> np.ndarray:
    pc = np.fromfile(bin_path, dtype=np.float32)
    # reshape into (x, y, z, reflectance)
    # reflectance = intensity = strength of LiDAR returned
    pc = pc.reshape(-1, 4)
    return pc

def log_pc(pc: np.ndarray):
    print("Pointcloud shape:", pc.shape)
    print("x/front-back range (meters):", pc[:,0].min(), pc[:,0].max())
    print("y/left-right range (meters):", pc[:,1].min(), pc[:,1].max())
    print("z/up-down range (meters):", pc[:,2].min(), pc[:,2].max())
    print("reflectance range:", pc[:,3].min(), pc[:,3].max())

if __name__ == "__main__":
    train_velo_path = "data/training/velodyne"
    lidar_files = sorted(glob.glob(train_velo_path + "/*.bin"))

    print(f"Found {len(lidar_files)} training LiDAR files")
    pc = load_velodyne_bin(lidar_files[0])
    log_pc(pc)
