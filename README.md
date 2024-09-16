# Fisheye_3d_Reconstruction
## About the project
The Fisheye_3d_Reconstruction project is a code challenge provided by [Teton](https://www.teton.ai/). It aims to process a video captured with a static fisheye camera and create 3D visualization about the static and moving objects.

![Results](doc/90_-45_-50_675.gif)

## Documentation
### Pipeline of the Solution
There are five steps in this solution:
1. Extract proper perspective images from the fisheye image sequence
2. Estimate the depth maps of the perspective images using pretrained monocular depth estimation model ([ZoeDepth](https://huggingface.co/docs/transformers/main/en/model_doc/zoedepth))
3. Apply a pretrained zero-shot text-conditioned object detection model ([OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)) to estimate the bounding boxes covering objects required attention, such as human, bed, chair, wheelchar etc.
4. Apply a pretrained semantic segmentation model ([SAM2](https://github.com/facebookresearch/segment-anything-2)) with the bounding boxes obtained in step 3 to estimate the semantic segmentation mask covering target objects
5. Convert depth map to 3D point cloud, filtered by segmentation semantic masks and colered by RGB perspective images

### Current Problem
1. The scale of depth maps estimated from different perspective views are largely different, making it tricky to stitch the point clouds from different perspective depth maps
2. The depth value of pixels far away from the camera are far larger than its real value, making part of the reconstructed point clouds highly distorted and stretched in the camera optical direction
3. The static point cloud in the scene could not be tracked consistently
4. The completeness of the object point clouds could be better, especially the moving human

### Explanation and Potential Solution
1. We need more prior information about the target scene, for example, a AprilTag or other visual fiducial system on the pillar, so that it can provide a reference for absolute scale. In addition, there tag indication will be beneficial for stitching multiple point clouds
2. The inaccuracy of depth value of remote pixels comes from the inner issue inherited from pixel image. As you can imagine, as the object further from the camera, the number of pixels representing the object in the pixel image is decresing, which is more ambiguous to estimate the accurate depth value. Therefore, I don't think it's a good idea or even possible to estimate good depth results for these area in the image  
3. The inconsistency of the static point cloud comes from the inconsistency of the object detection results - the bounding boxes. In order to increase the consistency of the estimated bounding boxes, we can apply some object tracking methods based on previous frames bounding boxes results
4. The incompleteness of the reconstructed point cloud comes from the fact that one pixel only represent one depth value along a light ray. In order to fix the hollow of the point cloud, one potential solution is by leveraging pretrained monocular RGB image to point cloud model 
    * end-to-end RGB to 3D structure, [RGB2Point](https://arxiv.org/pdf/2407.14979)
    * differentiable rendering RGB to 3D structure, [SoftRas](https://vgl.ict.usc.edu/Research/DMR/)
    * single image NeRF, [NerfDiff](https://arxiv.org/pdf/2302.10109)
    * image to 3D structure diffusion model, [PC^2](https://arxiv.org/pdf/2302.10668)

### TODO