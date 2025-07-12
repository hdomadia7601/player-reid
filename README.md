# Player Re-Identification — Sports Video Tracking 🏃‍♂️⚽

This project performs real-time **player tracking and re-identification** in sports footage using a fine-tuned **YOLOv8** detection model and **DeepSORT** for consistent identity assignment.

> **Objective:** Maintain consistent player IDs across a single video, even when players briefly leave and re-enter the frame.

---

## 🧠 Method Overview

- **Detection**: Fine-tuned YOLOv8 model trained to detect players and ball.
- **Tracking**: DeepSORT tracker maintains player identity across frames using Kalman filtering and appearance features.
- **Re-Identification**: When a player re-enters the frame, DeepSORT attempts to re-assign the correct ID using historical data and visual embeddings.
- **Visualization**: OpenCV is used to draw bounding boxes and IDs for each player across frames.

---

## 🔧 Project Structure
player-reid/
├── models/
│ └── yolo_model.pt # Fine-tuned YOLOv8 model (players + ball)
├── videos/
│ └── 15sec_input_720p.mp4 # Input video
├── outputs/
│ └── tracked_output.mp4 # Final video with tracked player IDs
├── src/
│ └── detect.py # Main tracking script
├── README.md
└── report.md


---

## 🚀 How to Run

### 1. Set up environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or manually
pip install ultralytics deep_sort_realtime opencv-python

2. Run the tracking script
python src/detect.py

✅ Output will be saved to:
outputs/tracked_output.mp4

Current Model Constraints
While the pipeline is functional and handles most scenarios well, certain constraints are observed due to the inherent limitations of DeepSORT and YOLOv8 in fast-paced sports contexts:

| Issue                           | Source         | Description                                                                              |
| ------------------------------- | -------------- | ---------------------------------------------------------------------------------------- |
| 🔁 Skipped IDs                  | Tracker        | When multiple players overlap or reappear rapidly, identity assignment may briefly fail. |
| 🧍 Missed Players               | Detector       | YOLOv8 may occasionally miss players due to occlusion or low confidence scores.          |
| 🔄 Duplicate IDs                | Tracker        | DeepSORT may reassign an existing ID to a different player after long occlusion gaps.    |
| 🔢 Non-sequential IDs           | Tracking Logic | ID numbers may appear non-linear as assigned by DeepSORT’s internal logic.               |
| 🔍 No Jersey Number Recognition | Scope          | This pipeline is purely visual — does not rely on jersey number OCR.                     |

These behaviors are expected from the current model choices and are typical of real-time tracking systems in visually dense sports footage.

Potential Enhancements (Next Steps)
The following improvements could increase tracking accuracy:

| Feature                      | Description                                                    | Tools/Models                       |
| ---------------------------- | -------------------------------------------------------------- | ---------------------------------- |
| 🧬 Re-ID with Embeddings     | Integrate OSNet or FastReID for better visual feature matching | `torchreid`, `FastReID`            |
| 🎯 Jersey Number OCR         | Detect and read jersey numbers for ID assignment               | `EasyOCR`, `Tesseract`, custom CNN |
| 🎥 Multi-Camera Support      | Re-identify players across different camera angles             | Spatial + appearance matching      |
| 📈 Evaluation Metrics        | Add IDF1, mAP, ID switches, etc.                               | `py-motmetrics`                    |
| 🎓 Fine-Tuned Sports Dataset | Improve accuracy by training detection on sport-specific data  | Roboflow, Label Studio             |
