from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    # ========= Input =========
    # ファイル入力: "../input_videos/5.mp4"
    # RTSP例: "rtsp://user:pass@host:554/stream"
    INPUT: str = "../input_videos/sample2.mov"
    IS_RTSP: bool = False  # RTSP等のストリーム入力なら True
    COOLDOWN_S: float = 1.0   # 1イベント保存後、すぐ新規開始しない


    # ========= Output =========
    OUTPUT_DIR: str = "../output/clips"

    # ========= Model =========
    WEIGHTS: str = "yolo_small_weights.pt"
    CLASS_ID: int = 1          # Violence/Fight
    CONF_THRES: float = 0.7
    IMG_SIZE: int = 640
    DEVICE: str = "0"          # "0"=CUDA:0, "cpu"=CPU
    HALF: bool = True          # 対応GPUならFP16推奨

    # ========= Speed =========
    FRAME_SKIP: int = 1        # Nフレームに1回推論（例:2なら2フレームに1回）
    MAX_FPS: float | None = None  # 入力が速すぎる場合の上限（Noneで制限なし）

    # ========= Offline segmentation =========
    MIN_EVENT_S: float = 1.0
    MERGE_GAP_S: float = 5.0 #検知が途切れてもこの秒数はイベント継続
    PAD_S: float = 0.7

    # ========= Realtime recording =========
    PREBUFFER_S: float = 1.0     # 検知開始前に遡って保存する秒数
    POSTBUFFER_S: float = 1.0    # 検知終了後に追加で保存する秒数
    END_HOLD_S: float = 5.0      # 検知が途切れてもこの秒数はイベント継続扱い

    # ========= Offline clipping =========
    USE_FFMPEG_COPY: bool = True   # True: -c copy（高速/キーフレームズレあり）
    FFMPEG_CRF: int = 23           # re-encode時
    FFMPEG_PRESET: str = "veryfast"

    # ========= Encoding for realtime async saver =========
    REALTIME_FOURCC: str = "mp4v"  # 環境により"avc1","H264","XVID"などに変更
