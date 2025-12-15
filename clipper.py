import os
import cv2
from segments import Segment
from utils import ensure_dir

def cut_segment_opencv(
    input_video: str,
    out_path: str,
    seg: Segment,
    fps: float,
) -> None:
    ensure_dir(os.path.dirname(out_path))

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {input_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # mp4で書き出す（環境によっては 'avc1' が通る場合もあります）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("VideoWriter を開けません。別のfourccや拡張子(.avi)を試してください。")

    start_f = max(0, int(round(seg.start_s * fps)))
    end_f = max(start_f, int(round(seg.end_s * fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    for _ in range(end_f - start_f + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    writer.release()
    cap.release()
