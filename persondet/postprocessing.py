"""
Post-processing utilities for YOLOX and CIGPose.
"""

import numpy as np


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in numpy."""
    # boxes: (N, 4), scores: (N, num_classes)
    # We assume single class (person) for now or handle multiclass

    # For YOLOX from rtmlib/mmpose, it usually exports (1, N, 4) and (1, N, C)
    # Or flattened.

    # Simple single-class NMS for person
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    # Filter by score
    mask = scores > score_thr
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    # Sort by score
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]

    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Current best is at index 0 (already sorted)

        # IoU
        rest_boxes = boxes[1:]

        x1 = np.maximum(boxes[0, 0], rest_boxes[:, 0])
        y1 = np.maximum(boxes[0, 1], rest_boxes[:, 1])
        x2 = np.minimum(boxes[0, 2], rest_boxes[:, 2])
        y2 = np.minimum(boxes[0, 3], rest_boxes[:, 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        inter = w * h

        area_cur = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1])
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])

        union = area_cur + area_rest - inter
        iou = inter / (union + 1e-6)

        # Keep those with IoU < threshold
        valid_mask = iou < nms_thr

        # Update for next it
        boxes = rest_boxes[valid_mask]
        scores = scores[1:][valid_mask]
        indices = indices[1:][valid_mask]

    # New attempt clean
    # Input: box [N, 4], score [N] (already filtered)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]

    keep_indices = []

    while order.size > 0:
        i = order[0]
        keep_indices.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return boxes[keep_indices], scores[keep_indices]


def simcc_decode(simcc_x, simcc_y, split_ratio=2.0):
    """Decode SimCC logits to keypoint coordinates and confidence scores.

    Returns (K, 2) keypoints in model-input space and (K,) scores.
    """
    x_locs = np.argmax(simcc_x[0], axis=-1).astype(np.float32)
    y_locs = np.argmax(simcc_y[0], axis=-1).astype(np.float32)
    scores = np.minimum(np.max(simcc_x[0], axis=-1), np.max(simcc_y[0], axis=-1))
    kpts = np.stack([x_locs / split_ratio, y_locs / split_ratio], axis=-1)
    return kpts, scores
