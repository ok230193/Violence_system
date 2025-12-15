import os

from config import AppConfig
from utils import ensure_dir
from detector import detect_hit_frames
from segments import frames_to_segments
from clipper import cut_segment_opencv

def main():
    cfg = AppConfig()
    ensure_dir(cfg.OUTPUT_DIR)

    hit_frames, fps = detect_hit_frames(
        video_path=cfg.INPUT_VIDEO,
        weights=cfg.WEIGHTS,
        class_id=cfg.CLASS_ID,
        conf_thres=cfg.CONF_THRES,
        imgsz=cfg.IMGSZ,
    )

    if len(hit_frames) < cfg.MIN_HITS:
        print("暴力検知区間が見つかりません（または検知が少なすぎます）")
        return

    segments = frames_to_segments(
        hit_frames=hit_frames,
        fps=fps,
        min_event_s=cfg.MIN_EVENT_S,
        merge_gap_s=cfg.MERGE_GAP_S,
        pad_s=cfg.PAD_S,
    )

    if not segments:
        print("区間化後に有効なイベントがありません（短すぎる等）")
        return

    base = os.path.splitext(os.path.basename(cfg.INPUT_VIDEO))[0]

    for i, seg in enumerate(segments, 1):
        out_name = f"{base}_violence_{i:02d}_{seg.start_s:.2f}-{seg.end_s:.2f}.mp4"
        out_path = os.path.join(cfg.OUTPUT_DIR, out_name)
        cut_segment_opencv(cfg.INPUT_VIDEO, out_path, seg, fps)
        print(f"saved: {out_path}")

    print(f"done. segments={len(segments)}")

if __name__ == "__main__":
    main()
