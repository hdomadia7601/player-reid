# Assignment Report — Player Re-Identification

## 👤 Applicant
**Het Domadia**  
Email: hetdomadia025@gmail.com  
---

## 🎯 Task Completed
**Option 2: Re-Identification in a Single Feed**

Track each player in a 15-second football video and ensure that each retains the same identity even after leaving and re-entering the frame.

---

## 🧠 Methodology

### 1. Detection
- Used the provided fine-tuned **YOLOv8** model (`yolo_model.pt`) for player and ball detection.
- Only class 0 (players) was used for re-identification.

### 2. Tracking
- Integrated `deep_sort_realtime` to assign and persist unique player IDs.
- The tracker uses motion + appearance features to maintain ID consistency.

### 3. Visualization
- Used OpenCV to draw bounding boxes and player labels (`Player <ID>`) on each frame.
- Final output video is saved as `outputs/tracked_output.mp4`.

---

## 🧱 Challenges

- No prior experience with YOLO or DeepSORT — had to learn basics on the go.
- Real-time video frame extraction, detection, and re-identification was completely new.
- Minor bugs while installing and referencing paths — resolved through experimentation.

---

## 🔮 Improvements (with more time)

- Better re-ID with jersey number or color embeddings.
- Improve ID persistence in cases of occlusion or tight player clusters.
- Expand to multi-camera (Option 1) using feature-based matching or tracklet linking.

---

## ✅ Summary

- Fully working pipeline for Option 2
- Player detection + re-identification achieved with minor issues with consistent IDs
- Code is modular, reproducible, and beginner-friendly
- Learned end-to-end basics of a real-world CV task from scratch

---

## 📎 Files Submitted

- `detect.py` (tracking script)
- `yolo_model.pt` (YOLOv8 weights)
- `15sec_input_720p.mp4` (input video)
- `tracked_output.mp4` (output)
- `README.md`, `report.md`

