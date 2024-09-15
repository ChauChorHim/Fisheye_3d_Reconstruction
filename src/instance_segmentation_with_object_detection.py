import os
import sys

from transformers import pipeline
from PIL import Image, ImageDraw
import torch
import numpy as np

sys.path.append("segment-anything-2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def nms(bboxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.
    
    Parameters:
    - bboxes (ndarray): An array of shape (N, 4) containing the bounding boxes in the format [x1, y1, x2, y2].
    - scores (ndarray): An array of shape (N,) containing the confidence scores for each bounding box.
    - iou_threshold (float): IoU threshold for suppressing overlapping boxes.
    
    Returns:
    - keep (list): Indices of bounding boxes to keep.
    """
    if len(bboxes) == 0:
        return []

    # Coordinates of bounding boxes
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # Compute the area of each bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by their corresponding scores (from high to low)
    order = scores.argsort()[::-1]

    keep = []  # List of indices of boxes to keep

    while order.size > 0:
        i = order[0]  # Index of the current highest score box
        keep.append(i)

        # Compute IoU of this box with the remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute the width and height of the overlapping area
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the Intersection over Union (IoU)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Keep boxes with IoU less than the threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]  # Update the order to only keep non-suppressed boxes

    return keep

class InstanceSegmentationModule:
    def __init__(self) -> None:
        self.object_detector = pipeline(model="checkpoints/owlv2-base-patch16-ensemble", task="zero-shot-object-detection", device="cuda")
        self.instance_predictor = SAM2ImagePredictor(
            build_sam2("sam2_hiera_l.yaml", 
                       "checkpoints/sam2_hiera_large.pt")
                       ) 
        
    def instance_segmentation(self, image, candidate_labels):
        return_dict = {"object_bbox": None, "instance_mask": []}

        bbox_predictions = self.object_detector(
            image,
            candidate_labels=candidate_labels,
        )
    
        draw = ImageDraw.Draw(image)
        label_score = {label: [] for label in candidate_labels}
        label_bbox = {label: [] for label in candidate_labels}

        for prediction in bbox_predictions:
            bbox = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]
            if score < 0.1:
                continue
            label_score[label].append(score)
            label_bbox[label].append(bbox)

        filtered_prediction = [] 
        for label in candidate_labels:
            scores = np.array(label_score[label])
            bboxes = np.array([[bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]] for bbox in label_bbox[label]])
            keep_idxes = nms(bboxes, scores, iou_threshold=0.01)
            for iidx, idx in enumerate(keep_idxes):
                cur_bbox = bboxes[idx]
                cur_score = scores[idx]

                if iidx > 0:
                    max_score = scores[keep_idxes[0]]
                    if  (max_score - cur_score) > cur_score * 0.5:
                        break

                filtered_prediction.append({"label": label, "box": cur_bbox, "score": cur_score})
    
    
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            cur_label = None
            cur_id = 0
            for prediction in filtered_prediction:
                bbox = prediction["box"]
                label = prediction["label"]
                score = prediction["score"]

                if cur_label != label:
                    cur_id = 0
                    cur_label = label
                else:
                    cur_id += 1

                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
                draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")
    
                self.instance_predictor.set_image(image)
                masks, scores, _ = self.instance_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array([xmin, ymin, xmax, ymax]),
                    multimask_output=False,
                )

                mask_PIL = Image.fromarray(masks[0] * 255)
                return_dict["instance_mask"].append({"mask": mask_PIL, "label": label, "score": score})

        return_dict["object_bbox"] = image

        return return_dict