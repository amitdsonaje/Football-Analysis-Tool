import numpy as np
import cv2

def track_ball(ball_centers, frame_size):
    """
    Given a list of (x, y) ball positions, draw the trajectory over a football pitch.
    """
    width, height = frame_size
    # Start with a green pitch
    pitch = np.ones((height, width, 3), dtype=np.uint8)
    pitch[:] = (0, 128, 0)  # dark green

    # Draw basic pitch lines
    pitch = draw_pitch_lines(pitch)

    # Draw ball trajectory (only if we have at least 2 points)
    for i in range(1, len(ball_centers)):
        if ball_centers[i - 1] is None or ball_centers[i] is None:
            continue
        pt1 = tuple(ball_centers[i - 1])
        pt2 = tuple(ball_centers[i])
        cv2.line(pitch, pt1, pt2, (0, 0, 255), 2)  # Red trajectory

    # Draw ball circles
    for point in ball_centers:
        if point is not None:
            cv2.circle(pitch, tuple(point), 4, (255, 255, 255), -1)  # white dot

    return pitch

def draw_pitch_lines(img):
    """
    Draw basic football pitch lines on the image.
    """
    height, width, _ = img.shape
    # Midline
    cv2.line(img, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
    # Center circle
    cv2.circle(img, (width // 2, height // 2), 70, (255, 255, 255), 2)
    # Penalty boxes
    cv2.rectangle(img, (0, height // 3), (100, 2 * height // 3), (255, 255, 255), 2)
    cv2.rectangle(img, (width - 100, height // 3), (width, 2 * height // 3), (255, 255, 255), 2)
    # Outer boundary
    cv2.rectangle(img, (5, 5), (width - 5, height - 5), (255, 255, 255), 2)
    return img
