from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import os

# Paths
video_path = "../videos/15sec_input_720p.mp4"
model_path = "models/yolo_model.pt"
output_path = "../outputs/tracked_output.mp4"

# Load YOLOv8 model
model = YOLO(model_path)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_num = 0
print("🔁 Starting tracking...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video ended.")
        break

    results = model(frame)[0]

    detections = []
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0:  # Only track class 0: players
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_num += 1
    print(f"✅ Frame {frame_num} processed.")

cap.release()
out.release()
print("\n✅ Player tracking complete! Output saved to:", output_path)

