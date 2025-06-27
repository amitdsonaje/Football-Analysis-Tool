# app.py
import streamlit as st
import cv2
import os
from detection import load_detection_model, detect_players_and_ball
from tracking import track_ball
from clustering import cluster_teams
from analysis import generate_heatmap, draw_voronoi
from config import ROBOFLOW_API_KEY

st.set_page_config(page_title="Football Analysis Tool", layout="wide")
st.title("Football Video Analysis")

# Upload video file
video_file = st.file_uploader("Upload a Football Video", type=["mp4", "mov", "avi"])
st.sidebar.title("Select Analysis Features")
features = st.sidebar.multiselect(
    "Enable features:", 
    ["Player Detection", "Ball Tracking", "Heatmap", "Trajectory Projection", "Team Clustering", "Voronoi Diagram"]
)

if video_file is not None:
    # Save uploaded video to disk
    tfile = open("input_video.mp4", "wb")
    tfile.write(video_file.read())
    tfile.close()
    st.success("Video uploaded successfully.")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Load detection model (Roboflow)
    model = load_detection_model(api_key=ROBOFLOW_API_KEY)

    # Initialize lists for analysis
    ball_centers = []
    player_positions = []

    cap = cv2.VideoCapture("input_video.mp4")
    frame_idx = 0
    annotated_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # if frame_idx % 5 != 0: continue

        # Perform detection
        detections = detect_players_and_ball(frame, model)
        st.text(f"Processing frame {frame_idx}")
        print(f"Frame {frame_idx} detections: {detections}")
        # Draw bounding boxes for players/ball if detection enabled
        if "Player Detection" in features or "Team Clustering" in features:
            print("Detected objects in frame:")
            for det in detections:
                print("Detection:", det)
                x, y, w, h = det["box"]  # det fields: box=(x,y,w,h), class, confidence
                cls = det["class"]
                # Save player centers for heatmap and clustering
                if cls == "player":
                    cx = x + w // 2
                    cy = y + h // 2
                    player_positions.append((cx, cy))
                # Save ball center for trajectory
                if cls == "ball":
                    cx = x + w // 2
                    cy = y + h // 2
                    ball_centers.append((cx, cy))
                # Draw bounding boxes on frame for visualization
                if cls == "player" and "Player Detection" in features:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, "Player", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                if cls == "ball" and "Player Detection" in features:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
                    cv2.putText(frame, "Ball", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    
        annotated_frames.append(frame)

    cap.release()

    # Save the last annotated frame for reference
    if annotated_frames:
        last_frame = annotated_frames[-1]
        cv2.imwrite("output/annotated_last_frame.jpg", last_frame)
        st.image(last_frame[:, :, ::-1], caption="Annotated Last Frame")

    # Ball Tracking (trajectory) -------------------------------------------------
    if "Ball Tracking" in features:
        print("Tracking ball trajectory with centers:", ball_centers)
        traj_img = track_ball(ball_centers, (last_frame.shape[1], last_frame.shape[0]))
        cv2.imwrite("output/ball_trajectory.jpg", traj_img)
        st.image(traj_img[:, :, ::-1], caption="Ball Trajectory")

    # Heatmap ---------------------------------------------------------------------
    if "Heatmap" in features:
        print("Generating heatmap with player positions:")
        heatmap_img = generate_heatmap(player_positions)
        cv2.imwrite("output/heatmap.png", heatmap_img)
        st.image(heatmap_img, caption="Player Heatmap")

    # Team Clustering -------------------------------------------------------------
    if "Team Clustering" in features:
        labels = cluster_teams(last_frame, detections)
        # Draw clusters on last_frame for visualization
        for det, label in zip(detections, labels):
            if det["class"] != "player":
                continue
            x, y, w, h = det["box"]
            color = (255,0,0) if label==0 else (0,0,255)
            cv2.rectangle(last_frame, (x, y), (x+w, y+h), color, 2)
            team_name = f"Team {label+1}"
            cv2.putText(last_frame, team_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite("output/team_clusters.jpg", last_frame)
        st.image(last_frame[:, :, ::-1], caption="Team Clustering")

    # Voronoi Diagram -------------------------------------------------------------
    if "Voronoi Diagram" in features:
        vor_img = draw_voronoi(player_positions, (last_frame.shape[1], last_frame.shape[0]))
        cv2.imwrite("output/voronoi.png", vor_img)
        st.image(vor_img[:, :, ::-1], caption="Voronoi Diagram")
