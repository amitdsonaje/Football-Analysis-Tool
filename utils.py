# utils.py
import numpy as np
import cv2

def apply_homography(points, src_points, dst_points):
    """
    Apply a homography to transform `points` from the image plane to another plane.
    `src_points` and `dst_points` are corresponding points defining the transform.
    """
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    M, status = cv2.findHomography(src, dst)
    if M is None:
        raise RuntimeError("Homography could not be computed.")
    points_np = np.array(points, dtype=np.float32).reshape(-1,1,2)
    transformed = cv2.perspectiveTransform(points_np, M)
    return transformed.reshape(-1, 2)
