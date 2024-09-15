import os
import math

import cv2
import numpy as np
import open3d as o3d
import glob


def correct_depth(depth_map, instance_map):
    # Apply a scaling factor to reduce depth for distant objects
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    depth_med = np.median(depth_map)
    depth_thres = (depth_max - depth_med) * 0.25 + depth_med

    # Adjust the depth map
    depth_map[depth_map > depth_thres] = 0

    return depth_map 


def rotate_z(pc, angle):
    angle = angle * np.pi / 180.0
    rotatation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                  [np.sin(angle), np.cos(angle), 0],
                                  [0, 0, 1]])
    pc.rotate(rotatation_matrix)
    return pc

def rotate_y(pc, angle):
    angle = angle * np.pi / 180.0
    rotatation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                  [0, 1, 0],
                                  [-np.sin(angle), 0, np.cos(angle)]])
    pc.rotate(rotatation_matrix)
    return pc

def rotate_x(pc, angle):
    angle = angle * np.pi / 180.0
    rotatation_matrix = np.array([[1, 0, 0],
                                  [0, np.cos(angle), -np.sin(angle)],
                                  [0, np.sin(angle), np.cos(angle)]])
    pc.rotate(rotatation_matrix)
    return pc

def compute_point_cloud_from_rgb_depth(rgb_image, depth_map, K):
    """
    Convert RGB and depth map to a point cloud.
    
    Parameters:
    - rgb_image: numpy array of shape (H, W, 3), the RGB image
    - depth_map: numpy array of shape (H, W), the depth map
    - K: numpy array of shape (3, 3), the intrinsic camera matrix
    
    Returns:
    - point_cloud: Open3D point cloud object
    """
    # Get image dimensions
    height, width = depth_map.shape

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(x, y)
    
    x_over_z = (u - cx) / fx
    y_over_z = (v - cy) / fy

    # Z = depth_map / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    Z = depth_map
    X = x_over_z * Z
    Y = y_over_z * Z

    # Stack X, Y, Z to get a point cloud in camera coordinate system
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # Get RGB values corresponding to the points
    rgb_flat = rgb_image.reshape(-1,
                                 3) / 255.0  # Normalize RGB values to [0, 1]

    rgb_flat = rgb_flat[points[:, 2] != 0] 
    points = points[points[:, 2] != 0]

    # Create an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()

    # Assign points and colors to the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb_flat)

    return point_cloud

def compute_point_cloud_from_rgb_depth_instance_mask(rgb_image, depth_map, path_to_instance_masks_list, K):
    # Get image dimensions
    height, width = depth_map.shape

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(x, y)
    
    x_over_z = (u - cx) / fx
    y_over_z = (v - cy) / fy

    # Z = depth_map / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    Z = depth_map
    X = x_over_z * Z
    Y = y_over_z * Z

    instance_mask = np.zeros_like(Z)

    for _, path in path_to_instance_masks_list.items():
        # if label == "floor":
        #     continue
        cur_instance_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
        # Z = correct_depth(Z, cur_instance_mask)
        instance_mask = np.logical_or(instance_mask, cur_instance_mask).astype(np.uint8)


    # Stack X, Y, Z to get a point cloud in camera coordinate system
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # Get RGB values corresponding to the points
    rgb_flat = rgb_image.reshape(-1,
                                 3) / 255.0  # Normalize RGB values to [0, 1]
    instance_mask = instance_mask.reshape(-1, )

    selected_idx = instance_mask == 1
    rgb_flat = rgb_flat[selected_idx] 
    points = points[selected_idx]

    # Create an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()

    # Assign points and colors to the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb_flat)

    return point_cloud

class Depth2PC:
    def __init__(self, K, width, height):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        x = np.linspace(0, width-1, width)
        y = np.linspace(0, height-1, height)

        # Create a meshgrid of pixel coordinates
        u, v = np.meshgrid(x, y)
    
        self.x_over_z = (u - cx) / fx
        self.y_over_z = (v - cy) / fy

    def rgb2pc(self, rgb_image, depth_map, instance_mask_list):

        Z = depth_map
        X = self.x_over_z * Z
        Y = self.y_over_z * Z

        instance_mask = np.zeros_like(Z)

        for cur_instance_mask in instance_mask_list:
            if np.max(cur_instance_mask) == 255:
                cur_instance_mask = np.array(cur_instance_mask) / 255
            elif np.max(cur_instance_mask) != 1:
                assert ValueError
            cur_instance_mask.astype(np.uint8)
            instance_mask = np.logical_or(instance_mask, cur_instance_mask).astype(np.uint8)

        # Stack X, Y, Z to get a point cloud in camera coordinate system
        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

        # Get RGB values corresponding to the points
        rgb_flat = rgb_image.reshape(-1,
                                     3) / 255.0  # Normalize RGB values to [0, 1]
        instance_mask = instance_mask.reshape(-1, )

        selected_idx = instance_mask == 1
        rgb_flat = rgb_flat[selected_idx] 
        points = points[selected_idx]

        # Create an Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()

        # Assign points and colors to the point cloud
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(rgb_flat)

        point_cloud = point_cloud.voxel_down_sample(0.03)

        # Apply statistical outlier removal
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
        point_cloud = point_cloud.select_by_index(ind)

        return point_cloud

if __name__ == "__main__":
    path_to_rgb = "results/persp"
    path_to_depth = "results/depth"
    path_to_output = "results/pointcloud"
    path_to_instance_mask = "results/instance_mask"
    os.makedirs("results/pointcloud", exist_ok=True)

    for root, dirs, files in os.walk(path_to_rgb):
        for image_path in glob.glob(os.path.join(root, "*.png")):

            file_name = os.path.basename(image_path)
            [fov, theta, phi] = [
                float(value)
                for value in file_name.replace(".png", "").split("_")
            ]

            rgb_image = cv2.imread(os.path.join(path_to_rgb, file_name),
                                   cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            depth_map = cv2.imread(os.path.join(path_to_depth, file_name),
                                   cv2.IMREAD_GRAYSCALE)
            # depth_map = correct_depth(depth_map)
            image_width, image_height = depth_map.shape

            # Get all instance masks
            current_instance_masks = os.path.join(path_to_instance_mask, file_name.replace(".png", "")) 
            path_to_instance_masks_list = {
                file.replace(".png", ""): os.path.join(current_instance_masks, file) 
                for file in os.listdir(current_instance_masks)
            }
            
            K = np.zeros((3, 3))
            K[0, 0] = 676
            K[1, 1] = 673
            K[0, 2] = 675
            K[1, 2] = 675
            K[2, 2] = 1

            # Smooth the depth map
            # depth_map = cv2.bilateralFilter(depth_map, 15, 75, 75)

            # pc = compute_point_cloud_from_rgb_depth(rgb_image, depth_map, K)
            pc = compute_point_cloud_from_rgb_depth_instance_mask(rgb_image, depth_map, path_to_instance_masks_list, K)

            pc = rotate_x(pc, -phi)
            # pc = rotate_z(pc, theta)

            # Voxel down sample
            pc = pc.voxel_down_sample(0.03)

            # Apply statistical outlier removal
            cl, ind = pc.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
            pc = pc.select_by_index(ind)

            # pc.estimate_normals()
            # pc.orient_normals_to_align_with_direction()

            o3d.io.write_point_cloud(
                os.path.join(path_to_output, file_name.replace(".png",
                                                               ".pcd")), pc)
