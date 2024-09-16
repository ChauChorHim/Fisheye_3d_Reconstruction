import os

from PIL import Image
import cv2
import torch
from transformers import pipeline
import numpy as np
import open3d as o3d
from tqdm import tqdm

from src.fisheye_mapper import Fish2Persp
from src.instance_segmentation_with_object_detection import InstanceSegmentationModule
from src.depth2pointcloud import Depth2PC


def estimate_depth(cur_image, depth_estimator):
    cur_depth = depth_estimator(cur_image)["depth"]

    ori_width, ori_height = cur_image.width, cur_image.height
    new_width, new_height = int(ori_width * 1.2), int(ori_height * 1.2)
    cur_depth = cur_depth.resize((new_width, new_height), Image.NEAREST)
    left = (cur_depth.width - ori_width) / 2
    right = (cur_depth.width + ori_width) / 2
    top = (cur_depth.height - ori_height) / 2
    bottom = (cur_depth.height + ori_height) / 2
    cur_depth = cur_depth.crop((left, top, right, bottom))
    return cur_depth

if __name__ == "__main__":

    # Define the perspective angles
    fov = 90
    theta = -45
    phi = -50
    focal = 675
    persp_shape = [928, 928]

    # Define the camera intrinsic matrix
    K = np.zeros((3, 3))
    K[0, 0] = 676
    K[1, 1] = 673
    K[0, 2] = int(persp_shape[0] / 2)
    K[1, 2] = int(persp_shape[1] / 2)
    K[2, 2] = 1

    fish2persp = Fish2Persp(persp_shape=persp_shape,
                                        theta=theta,
                                        phi=phi,
                                        fov=fov)
    # Prepare output folders
    path_to_fisheye_frames = "data/fisheye-3d-challenge/data/frames/"
    path_to_output = f"results/{fov}_{theta}_{phi}_{focal}"
    path_to_perspective = os.path.join(path_to_output, "persp")
    path_to_depth = os.path.join(path_to_output, "depth")
    path_to_pointcloud = os.path.join(path_to_output, "pointcloud")
    path_to_object_bbox = os.path.join(path_to_output, "object_bbox")
    path_to_instance_mask = os.path.join(path_to_output, "instance_mask")

    os.makedirs(path_to_perspective, exist_ok=True)
    os.makedirs(path_to_depth, exist_ok=True)
    os.makedirs(path_to_pointcloud, exist_ok=True)
    os.makedirs(path_to_object_bbox, exist_ok=True)
    os.makedirs(path_to_instance_mask, exist_ok=True)

    # Load monocular depth model and define the function for estimating the depth
    depth_estimator = pipeline(task="depth-estimation", model="checkpoints/zoedepth-nyu", device="cuda")

    # Load instance segmentation model with object detection model
    instance_seg = InstanceSegmentationModule()
    candidate_labels = ["floor", "mannequin", "bed", "chair", "white bin", "desk", "wheelchair", "pillar", "human", "cardboard boxes", "whiteboard"]

    # Load depth to pointcloud module
    depth2pc = Depth2PC(K, width=persp_shape[0], height=persp_shape[1])

    # Some controller
    save_persp = False
    save_depth = False
    save_bbox = True
    save_instance_mask = False
    save_pc = True
    start_from_idx = 80

    fisheye_frame_files = os.listdir(path_to_fisheye_frames)
    fisheye_frame_files.sort()
    fisheye_frame_files = fisheye_frame_files[start_from_idx:-1]

    for cur_frame_file in tqdm(fisheye_frame_files):

        cur_fisheye_image = cv2.imread(os.path.join(path_to_fisheye_frames, cur_frame_file), cv2.IMREAD_COLOR)
        cur_fisheye_image = cv2.cvtColor(cur_fisheye_image, cv2.COLOR_BGR2RGB)
        cur_fisheye_image = torch.from_numpy(cur_fisheye_image).float().permute(2, 0, 1).unsqueeze(0)
        
        # Get and save the perspective image
        cur_persp = fish2persp(cur_fisheye_image.cuda())
        cur_persp = (cur_persp.cpu().squeeze(0).permute(1,2,0).numpy()).astype(np.uint8).copy()
        cur_persp_pil = Image.fromarray(cur_persp)

        if save_persp:
            cur_persp_pil.save(os.path.join(path_to_perspective, cur_frame_file))

        # Get and save the depth map
        cur_depth = estimate_depth(cur_persp_pil, depth_estimator)

        if save_depth:
            cur_depth.save(os.path.join(path_to_depth, cur_frame_file)) 

        # Get and save the bounding box image and instance masks
        instance_seg_dict = instance_seg.instance_segmentation(cur_persp_pil, candidate_labels)

        if save_bbox:
            instance_seg_dict["object_bbox"].save(os.path.join(path_to_object_bbox, cur_frame_file))

        if save_instance_mask:
            cur_label = None
            cur_id = 0
            for instance_mask in instance_seg_dict["instance_mask"]:
                mask_image = instance_mask["mask"]
                mask_label = instance_mask["label"]
                if cur_label != mask_label:
                    cur_id = 0
                    cur_label = mask_label
                else:
                    cur_id += 1  
                path_to_current_instance_mask = os.path.join(path_to_instance_mask, cur_frame_file.replace(".png", ""))
                os.makedirs(path_to_current_instance_mask, exist_ok=True)
                mask_image = mask_image.convert("RGB")
                mask_image.save(os.path.join(path_to_current_instance_mask, f"{cur_label}-{cur_id}.png"))
        
        # Compute and save the pointcloud
        instance_mask_list = [np.array(instance_mask["mask"]) for instance_mask in instance_seg_dict["instance_mask"]]
        cur_pc = depth2pc.rgb2pc(cur_persp, np.array(cur_depth), instance_mask_list)
        
        if save_pc:
            o3d.io.write_point_cloud(
                    os.path.join(path_to_pointcloud, cur_frame_file.replace(".png", ".pcd")), cur_pc)