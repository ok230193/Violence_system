from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Deque, List, Optional, Tuple

import cv2

from clipper import (
    SinglePassClipJob,
    SinglePassClipper,
    cut_segment_ffmpeg_copy,
    cut_segment_ffmpeg_reencode,
)
from config import AppConfig
from detector import ViolenceFrameDetector, detect_hit_frames_offline
from segments import frames_to_segments
from utils import ensure_dir, FPSLimiter


@dataclass
class SaveJob:
    frames: List
    fps: float
    size: Tuple[int, int]  # (w,h)
    out_path: str
    fourcc: str


class AsyncSaver(Thread):
    """Write clips on a background thread so capture/inference stays realtime."""
    def __init__(self, q: "Queue[Optional[SaveJob]]"):
        super().__init__(daemon=True)
        self.q = q

    def run(self) -> None:
        while True:
            job = self.q.get()
            if job is None:
                return
            ensure_dir(os.path.dirname(job.out_path))
            clip = SinglePassClipper(
                SinglePassClipJob(
                    out_path=job.out_path,
                    fps=job.fps,
                    frame_size=job.size,
                    fourcc=job.fourcc,
                )
            )
            for f in job.frames:
                clip.write(f)
            clip.close()
            self.q.task_done()


def offline_file_mode(cfg: AppConfig) -> None:
    """File input: detect -> segments -> ffmpeg cut (fast)."""
    ensure_dir(cfg.OUTPUT_DIR)

    hit_frames, fps = detect_hit_frames_offline(
        video_path=cfg.INPUT,
        weights=cfg.WEIGHTS,
        class_id=cfg.CLASS_ID,
        conf_thres=cfg.CONF_THRES,
        imgsz=cfg.IMG_SIZE,
        device=cfg.DEVICE,
        half=cfg.HALF,
        frame_skip=max(1, cfg.FRAME_SKIP),
    )

    segments = frames_to_segments(
        hit_frames=hit_frames,
        fps=fps,
        min_event_s=cfg.MIN_EVENT_S,
        merge_gap_s=cfg.MERGE_GAP_S,
        pad_s=cfg.PAD_S,
    )

    if not segments:
        print("検知イベントなし")
        return

    base = os.path.splitext(os.path.basename(cfg.INPUT))[0]
    for i, seg in enumerate(segments, 1):
        out_name = f"{base}_violence_{i:02d}_{seg.start_s:.2f}-{seg.end_s:.2f}.mp4"
        out_path = os.path.join(cfg.OUTPUT_DIR, out_name)

        if cfg.USE_FFMPEG_COPY:
            cut_segment_ffmpeg_copy(cfg.INPUT, out_path, seg)
        else:
            cut_segment_ffmpeg_reencode(
                cfg.INPUT, out_path, seg, crf=cfg.FFMPEG_CRF, preset=cfg.FFMPEG_PRESET
            )
        print(f"saved: {out_path}")

    print(f"done. segments={len(segments)}")


def realtime_mode(cfg: AppConfig) -> None:
    """
    Single-pass + ring buffer + async save.

    - Read continuously
    - Keep PREBUFFER_S frames in deque
    - Infer every FRAME_SKIP frames
    - When detected starts: create clip with prebuffer
    - End when detection absent for END_HOLD_S, plus POSTBUFFER_S tail
    """
    ensure_dir(cfg.OUTPUT_DIR)

    cap = cv2.VideoCapture(cfg.INPUT)
    if not cap.isOpened():
        raise RuntimeError(f"入力を開けません: {cfg.INPUT}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0
    fps = float(fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    detector = ViolenceFrameDetector(
        weights=cfg.WEIGHTS,
        class_id=cfg.CLASS_ID,
        conf_thres=cfg.CONF_THRES,
        imgsz=cfg.IMG_SIZE,
        device=cfg.DEVICE,
        half=cfg.HALF,
    )

    prebuf_len = max(1, int(max(0.0, cfg.PREBUFFER_S) * fps))
    postbuf_len = int(max(0.0, cfg.POSTBUFFER_S) * fps)
    ring: Deque = deque(maxlen=prebuf_len)

    q: "Queue[Optional[SaveJob]]" = Queue(maxsize=8)
    saver = AsyncSaver(q)
    saver.start()

    limiter = FPSLimiter(cfg.MAX_FPS)

    active_frames: List = []
    active = False
    post_remaining = 0
    last_detect_t = 0.0
    clip_count = 0

    frame_idx = 0
    while True:
        limiter.sleep_if_needed()

        ok, frame = cap.read()
        if not ok:
            break

        if w == 0 or h == 0:
            h, w = frame.shape[:2]

        ring.append(frame)

        detected = False
        if frame_idx % max(1, cfg.FRAME_SKIP) == 0:
            detected = detector.infer(frame)
            if detected:
                last_detect_t = time.time()

        now = time.time()
        holding = (now - last_detect_t) <= cfg.END_HOLD_S

        if not active:
            if detected:
                active = True
                post_remaining = postbuf_len
                active_frames = list(ring) + [frame]
                clip_count += 1
                print(f"[start] clip #{clip_count}")
        else:
            active_frames.append(frame)

            if holding:
                post_remaining = postbuf_len
            else:
                post_remaining -= 1
                if post_remaining <= 0:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_name = f"realtime_violence_{ts}_{clip_count:04d}.mp4"
                    out_path = os.path.join(cfg.OUTPUT_DIR, out_name)

                    job = SaveJob(
                        frames=active_frames,
                        fps=fps,
                        size=(w, h),
                        out_path=out_path,
                        fourcc=cfg.REALTIME_FOURCC,
                    )
                    try:
                        q.put_nowait(job)
                        print(f"[enqueue] {out_path} frames={len(active_frames)}")
                    except Exception:
                        print("[warn] saver queue full -> drop clip")

                    active = False
                    active_frames = []
                    post_remaining = 0
                    last_detect_t = 0.0

        frame_idx += 1

    cap.release()
    q.put(None)
    print("realtime done.")


def main():
    cfg = AppConfig()
    if cfg.IS_RTSP:
        realtime_mode(cfg)
    else:
        offline_file_mode(cfg)


if __name__ == "__main__":
    main()
