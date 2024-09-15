import time
from pathlib import Path
import os

import numpy as onp
import tyro
from tqdm.auto import tqdm
import open3d as o3d

import viser
import viser.extras
import viser.transforms as tf


def main(
    data_path: Path = Path(__file__).parent / "/root/Fisheye_3d_Reconstruction/results/90_-45_-50_675/pointcloud",
    max_frames: int = 100,
    share: bool = False,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading frames!")
    loader = []
    for file in os.listdir(data_path):
        cur_pc = o3d.io.read_point_cloud(os.path.join(data_path, file))
        loader.append(cur_pc)
    num_frames = min(max_frames, len(loader))

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )

    frame_nodes: list[viser.FrameHandle] = []
    for i in tqdm(range(num_frames)):
        frame = loader[i]
        position, color = frame.points, frame.colors 

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=onp.asarray(position),
            colors=onp.asarray(color),
            point_size=0.01,
            point_shape="rounded",
        ) 

    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = False

    # Playback update loop.
    while True:
        for i, frame_node in enumerate(frame_nodes):
            frame_node.visible = True
            time.sleep(1.0 / 5)
            frame_node.visible = False

if __name__ == "__main__":
    tyro.cli(main)