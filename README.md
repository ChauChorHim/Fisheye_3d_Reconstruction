# Fisheye_3d_Reconstruction
## About the project
The Fisheye_3d_Reconstruction project is a code challenge provided by [Teton](https://www.teton.ai/). It aims to process a video captured with a static fisheye camera and create 3D visualization about the static and moving objects.

![Results](doc/Recording 2024-09-15 215428.gif)

## Documentation
### Pipeline of the Solution
There are five steps in this solution:
1. Extract proper perspective images from the fisheye image sequence
2. Estimate the depth maps of the perspective images using pretrained monocular depth estimation model ([ZoeDepth](https://huggingface.co/docs/transformers/main/en/model_doc/zoedepth))
3. Apply a pretrained zero-shot text-conditioned object detection model ([OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)) to estimate the bounding boxes covering objects required attention, such as human, bed, chair, wheelchar etc.
4. Apply a pretrained semantic segmentation model ([SAM2](https://github.com/facebookresearch/segment-anything-2)) with the bounding boxes obtained in step 3 to estimate the semantic segmentation mask covering target objects
5. Convert depth map to 3D point cloud, filtered by segmentation semantic masks and colered by RGB perspective images