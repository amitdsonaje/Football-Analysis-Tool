# ⚽ Football Match Analysis Tool
  This project is an end-to-end football (soccer) video analysis tool built with Python and a user-friendly Streamlit interface. It allows users to upload a match video and choose from various real-time analytics features to better understand player and ball behavior on the pitch.

Features
1. Player Detection – Using a pre-trained Roboflow object detection model via API.
2. Ball Tracking & Trajectory Visualization – Tracks the ball across frames and draws its movement path.
3. Player Heatmap – Shows areas of maximum player presence across the match.
4. Team Clustering – Separates players into two teams based on jersey color using K-Means clustering.
5. Interactive GUI – Built with Streamlit for easy use without coding knowledge.
6. Output Saving – Saves annotated frames and analysis visuals to a local output/ folder.

Tech Stack
Python 3.8+
Streamlit – Interactive UI
OpenCV – Image processing
Roboflow – Pre-trained detection models via API
NumPy & Matplotlib – Array and plotting tools
scikit-learn – K-Means clustering
