# clustering.py
import cv2
import numpy as np
from sklearn.cluster import KMeans

def cluster_teams(frame, detections):
    """
    Cluster detected players into two teams based on jersey color.
    Returns a list of cluster labels (0 or 1) corresponding to each player detection.
    """
    player_colors = []
    player_indices = []
    for idx, det in enumerate(detections):
        if det["class"] != "player":
            continue
        x, y, w, h = det["box"]
        # Crop the player bounding box from the frame
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        # Convert to HSV and take hue channel to reduce lighting effect (optional)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Compute average hue value as feature
        mean_hue = np.mean(roi_hsv[:, :, 0])
        player_colors.append([mean_hue])
        player_indices.append(idx)
    labels = []
    if len(player_colors) >= 2:
        # K-Means on average jersey color (2 clusters: two teams)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(player_colors))
        labels_all = kmeans.labels_
        # Map back to detection order
        idx_to_label = dict(zip(player_indices, labels_all))
        for i, det in enumerate(detections):
            if det["class"] == "player":
                labels.append(idx_to_label.get(i, 0))
            else:
                labels.append(None)
    else:
        # Not enough players to cluster; assign default
        for det in detections:
            labels.append(0 if det["class"]=="player" else None)
    return labels
