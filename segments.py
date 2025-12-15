from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Segment:
    start_s: float
    end_s: float

def frames_to_segments(
    hit_frames: List[int],
    fps: float,
    min_event_s: float,
    merge_gap_s: float,
    pad_s: float,
) -> List[Segment]:
    if not hit_frames:
        return []

    hit_frames = sorted(set(hit_frames))

    # 連続フレームを区間化（frame index）
    raw: List[Tuple[int, int]] = []
    s = hit_frames[0]
    prev = hit_frames[0]
    for f in hit_frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            raw.append((s, prev))
            s = f
            prev = f
    raw.append((s, prev))

    segs = [Segment(a / fps, b / fps) for a, b in raw]

    # ギャップが短いものは結合
    merged: List[Segment] = []
    for seg in segs:
        if not merged:
            merged.append(seg)
            continue
        if seg.start_s - merged[-1].end_s <= merge_gap_s:
            merged[-1].end_s = max(merged[-1].end_s, seg.end_s)
        else:
            merged.append(seg)

    # 前後パディング & 短すぎる区間は除外
    out: List[Segment] = []
    for seg in merged:
        start = max(0.0, seg.start_s - pad_s)
        end = seg.end_s + pad_s
        if (end - start) >= min_event_s:
            out.append(Segment(start, end))

    return out
