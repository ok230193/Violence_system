from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    # 入力動画
    INPUT_VIDEO: str = "../input_videos/5.mp4"

    # 出力フォルダ
    OUTPUT_DIR: str = "../output/clips"

    # モデル
    WEIGHTS: str = "yolo_small_weights.pt"
    CLASS_ID: int = 1          # Violence/Fight
    CONF_THRES: float = 0.4
    IMGSZ: int = 640

    # 区間生成の安定化
    MIN_HITS: int = 5
    MIN_EVENT_S: float = 1.0
    MERGE_GAP_S: float = 2.0
    PAD_S: float = 1.0

    # 切り出し
    REENCODE: bool = False     # 精密に切りたいなら True
