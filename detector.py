from typing import List
import cv2
import numpy as np
from ultralytics import YOLO

def detect_hit_frames(
    video_path: str,
    weights: str,
    class_id: int,
    conf_thres: float,
    imgsz: int,
) -> tuple[List[int], float]:
    model = YOLO(weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hit_frames: List[int] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, imgsz=imgsz, conf=conf_thres, verbose=False)
        r = results[0]

        found = False
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)
            confs = r.boxes.conf.detach().cpu().numpy()
            if np.any((cls == class_id) & (confs >= conf_thres)):
                found = True

        if found:
            hit_frames.append(frame_idx)

        frame_idx += 1

    cap.release()
    return hit_frames, fps
