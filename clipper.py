"""
Clipping utilities.

1) cut_segment_ffmpeg_copy: fastest for file inputs (stream copy, no re-encode)
2) cut_segment_ffmpeg_reencode: precise cut (re-encode)
3) SinglePassClipper: used for realtime / single-pass writing
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import cv2

from segments import Segment
from utils import ensure_dir


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")


def cut_segment_ffmpeg_copy(input_video: str, out_path: str, seg: Segment) -> None:
    """Fastest cut: stream copy (-c copy). Keyframe boundary may cause small drift."""
    ensure_dir(os.path.dirname(out_path))
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg が見つかりません。PATHにffmpegを入れてください。")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{seg.start_s:.3f}",
        "-to", f"{seg.end_s:.3f}",
        "-i", input_video,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    _run(cmd)


def cut_segment_ffmpeg_reencode(
    input_video: str,
    out_path: str,
    seg: Segment,
    crf: int = 23,
    preset: str = "veryfast",
) -> None:
    """Frame-accurate cut by re-encoding (slower)."""
    ensure_dir(os.path.dirname(out_path))
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg が見つかりません。PATHにffmpegを入れてください。")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{seg.start_s:.3f}",
        "-to", f"{seg.end_s:.3f}",
        "-i", input_video,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]
    _run(cmd)


@dataclass
class SinglePassClipJob:
    out_path: str
    fps: float
    frame_size: tuple[int, int]  # (w, h)
    fourcc: str = "mp4v"


class SinglePassClipper:
    """Open a VideoWriter once and write frames; used by realtime async saver."""
    def __init__(self, job: SinglePassClipJob):
        ensure_dir(os.path.dirname(job.out_path))
        w, h = job.frame_size
        self.writer = cv2.VideoWriter(
            job.out_path,
            cv2.VideoWriter_fourcc(*job.fourcc),
            job.fps,
            (w, h),
        )
        if not self.writer.isOpened():
            raise RuntimeError(
                f"VideoWriter を開けません: {job.out_path} (fourcc={job.fourcc}). "
                "別のfourccや拡張子(.avi)を試してください。"
            )

    def write(self, frame) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
