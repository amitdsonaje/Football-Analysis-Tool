# analysis.py
import numpy as np
import cv2
from scipy.spatial import Voronoi, voronoi_plot_2d

# analysis.py
import numpy as np
import cv2

def generate_heatmap(player_positions, field_size=(720, 1280), bins=100):
    """
    Generates a smooth heatmap overlayed on a football pitch background.
    - field_size: tuple (height, width) of the output image.
    - bins: resolution of the heatmap grid.
    """
    height, width = field_size
    heatmap_img = np.zeros((height, width, 3), dtype=np.uint8)

    if not player_positions:
        return heatmap_img

    # Convert positions to separate x and y lists
    x_coords = [p[0] for p in player_positions]
    y_coords = [p[1] for p in player_positions]

    # Generate 2D histogram
    heatmap, xedges, yedges = np.histogram2d(y_coords, x_coords, bins=bins, range=[[0, height], [0, width]])

    # Normalize the heatmap to 0–255
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)

    # Apply color map (JET = heat-like: blue → red)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize to match full field size
    heatmap_colored = cv2.resize(heatmap_colored, (width, height), interpolation=cv2.INTER_CUBIC)

    # Draw basic pitch lines for orientation
    pitch = draw_pitch_background(height, width)

    # Blend heatmap onto pitch
    overlay = cv2.addWeighted(pitch, 0.7, heatmap_colored, 0.3, 0)

    return overlay

def draw_pitch_background(height, width):
    # Green background
    pitch = np.ones((height, width, 3), dtype=np.uint8)
    pitch[:] = (0, 128, 0)  # Dark green BGR

    # Draw pitch lines in white
    cv2.line(pitch, (width//2, 0), (width//2, height), (255,255,255), 2)        # Midline
    cv2.circle(pitch, (width//2, height//2), 70, (255,255,255), 2)             # Center circle
    cv2.rectangle(pitch, (0, height//3), (100, 2*height//3), (255,255,255), 2) # Left penalty box
    cv2.rectangle(pitch, (width-100, height//3), (width, 2*height//3), (255,255,255), 2) # Right penalty
    cv2.rectangle(pitch, (5, 5), (width-5, height-5), (255,255,255), 2)        # Outer boundary
    return pitch


def draw_voronoi(points, frame_size):
    """
    Draw a Voronoi diagram for the given player points.
    Returns an RGB image of the same size with Voronoi regions drawn.
    """
    width, height = frame_size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(points) < 2:
        return img
    pts = np.array(points)
    vor = Voronoi(pts)
    # Draw Voronoi edges
    fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='white',
                          line_width=2, line_alpha=0.6, point_size=0)
    # Extract lines from matplotlib figure
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.destroyAllWindows()
    # Overlay original points
    for (x, y) in points:
        cv2.circle(img, (int(x), int(y)), radius=5, color=(0,255,0), thickness=-1)
    return img
