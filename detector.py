from __future__ import annotations

from typing import List

import cv2
import numpy as np
from ultralytics import YOLO


def _get_fps(cap: cv2.VideoCapture, fallback: float = 30.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        return fallback
    return float(fps)


def detect_hit_frames_offline(
    video_path: str,
    weights: str,
    class_id: int,
    conf_thres: float,
    imgsz: int,
    device: str = "0",
    half: bool = True,
    frame_skip: int = 1,
) -> tuple[List[int], float]:
    """Offline: read entire file and return hit frame indices."""
    model = YOLO(weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    fps = _get_fps(cap)
    hit_frames: List[int] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % max(1, frame_skip) == 0:
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf_thres,
                device=device,
                half=half,
                verbose=False,
            )
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


class ViolenceFrameDetector:
    """Realtime detector: infer(frame)->bool."""
    def __init__(
        self,
        weights: str,
        class_id: int,
        conf_thres: float,
        imgsz: int,
        device: str = "0",
        half: bool = True,
    ):
        self.model = YOLO(weights)
        self.class_id = class_id
        self.conf_thres = conf_thres
        self.imgsz = imgsz
        self.device = device
        self.half = half

    def infer(self, frame) -> bool:
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return False
        cls = r.boxes.cls.detach().cpu().numpy().astype(int)
        confs = r.boxes.conf.detach().cpu().numpy()
        return bool(np.any((cls == self.class_id) & (confs >= self.conf_thres)))
