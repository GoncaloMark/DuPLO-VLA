import zarr
import imageio
import numpy as np

# Load the Zarr file
z = zarr.open("/home/lnxdre4d/Desktop/3D-Diffusion-Policy/data/metaworld_pick-place_expert.zarr", mode='r')
imgs = z['data']['img'][:]  # shape: (T, H, W, C)

# Save as video
video_path = "expert_demo_fixed.mp4"
with imageio.get_writer(video_path, fps=30) as writer:
    for img in imgs:
        writer.append_data(img)

print(f"Video saved at {video_path}")
