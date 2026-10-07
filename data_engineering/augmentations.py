"""Tăng cường dữ liệu (Augmentation) & Cân bằng lớp (Class Balancing) cho Train.

Mục đích
--------
Giai đoạn 1 - Data Engineering của dự án Edge AI Object Detection cho xe RC.
Script này chỉ làm việc trên tập ``data/processed/Train`` (đầu ra của
``clean_and_filter.py``) và thực hiện hai việc:

1. Định nghĩa một pipeline ``albumentations`` mô phỏng "vật lý" của xe RC chạy
   trên sa bàn: ánh sáng thay đổi (Color Jitter / HSV shift), camera rung lắc
   khi xe chạy nhanh (Motion Blur), góc nhìn thay đổi khi xe ôm cua / nảy xóc
   (Random Affine), cộng thêm nhiễu cảm biến và nén JPEG.
2. Sinh thêm ảnh + nhãn YOLO (.txt) cho các lớp THIỂU SỐ để cân bằng số lượng
   bounding box giữa 10 lớp 0..9 (9 biển báo + 1 người đi bộ).

QUY ƯỚC QUAN TRỌNG (đã thống nhất với giảng viên)
--------------------------------------------------
* GIỮ NGUYÊN class id gốc 0..9, TUYỆT ĐỐI KHÔNG re-index. Box nào có id ngoài
  0..9 (không nên còn sau bước clean) sẽ bị bỏ khỏi ảnh sinh ra và được thống kê.
* Chỉ tăng cường tập Train. Val/Test phải giữ nguyên phân phối "thật" để đánh
  giá mô hình trung thực (xem giải thích ở ``_ensure_train_dir``).
* Mosaic KHÔNG làm offline ở đây: Ultralytics trainer đã áp dụng Mosaic ONLINE
  (``mosaic=1.0`` trong cấu hình huấn luyện), mỗi epoch ghép khác nhau nên đa
  dạng hơn nhiều so với lưu cố định ra đĩa.
* Script này cố ý KHÔNG lật ảnh (xem ``get_train_transform``). Khi train bằng
  Ultralytics PHẢI đặt ``fliplr=0.0`` (mặc định 0.5 sẽ lật ngang ONLINE một nửa
  số ảnh, phá vỡ lý do trên) và nên giữ ``hsv_h`` nhỏ (mặc định 0.015) vì nó
  CỘNG DỒN với phép dịch hue ở đây. Gợi ý được ghi vào báo cáo JSON
  (``recommended_train_hyp``).

Ảnh sinh ra có tên ``<stem>_aug<NNN>.jpg`` (+ ``<stem>_aug<NNN>.txt``). Mỗi lần
chạy, các file ``_aug`` cũ bị xóa trước nên chạy lại nhiều lần KHÔNG bị cộng dồn
(idempotent); cùng dữ liệu + cùng ``--seed`` cho ra kết quả giống hệt nhau.
Một file chỉ được coi là "do script sinh ra" khi tên có đuôi ``_augNNN`` VÀ ảnh
gốc ``<stem>`` tương ứng có mặt trong ``images/`` - ảnh gốc tình cờ có tên
dạng ``..._aug001`` (không có "cha") sẽ không bị xóa nhầm.

Cách dùng
---------
    # Cân bằng mọi lớp lên bằng lớp đông nhất (mặc định)
    python data_engineering/augmentations.py

    # Mục tiêu = 90% lớp đông nhất, mỗi ảnh gốc được nhân bản tối đa 3 lần
    python data_engineering/augmentations.py --target-ratio 0.9 --max-copies 3

    # Mục tiêu cố định 4000 box/lớp, giới hạn tối đa 2000 ảnh sinh thêm
    python data_engineering/augmentations.py --target 4000 --max-new-images 2000

    # Chỉ xuất ảnh minh họa pipeline (không ghi gì vào Train)
    python data_engineering/augmentations.py --preview-only

    # Xóa toàn bộ ảnh _aug đã sinh, đưa Train về trạng thái gốc
    python data_engineering/augmentations.py --clean-only

Thư viện: Python stdlib, opencv-python, numpy, pyyaml, albumentations
(tương thích cả albumentations 1.4.x và 2.0.x).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import albumentations as A
import cv2
import numpy as np
import yaml
from albumentations.core.composition import BaseCompose

# ---------------------------------------------------------------------------
# Hằng số & đường dẫn mặc định
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "Train"
DEFAULT_PREVIEW_PATH = PROJECT_ROOT / "reports" / "augmentation_preview.jpg"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEEP_CLASS_IDS: tuple[int, ...] = tuple(range(10))  # 0..9 - giữ nguyên id gốc
NUM_CLASSES = len(KEEP_CLASS_IDS)

# Tên lớp dự phòng (lấy từ data.yaml gốc của Roboflow) nếu không đọc được
# data/processed/dataset_filtered.yaml.
DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "Crosswalk sign",
    1: "Highway entrance sign",
    2: "Highway exit sign",
    3: "No-entry road sign",
    4: "One-way road sign",
    5: "Parking sign",
    6: "Priority sign",
    7: "Roundabout sign",
    8: "Stop sign",
    9: "Pedestrian",
}

AUG_TAG = "_aug"
AUG_STEM_RE = re.compile(r"_aug\d{3,}$")  # nhận diện file do script này sinh ra
NON_TRAIN_DIR_NAMES = {"val", "valid", "validation", "test", "testing"}

# Siêu tham số Ultralytics nên dùng cùng dữ liệu này (ghi vào báo cáo JSON):
# tắt lật ảnh online (lý do: xem get_train_transform), Mosaic làm online.
RECOMMENDED_TRAIN_HYP: dict[str, float] = {
    "fliplr": 0.0, "flipud": 0.0, "mosaic": 1.0, "hsv_h": 0.015,
}

# Dung sai khi kiểm tra tọa độ YOLO đọc từ file (tràn số thực nhỏ của Roboflow).
COORD_TOLERANCE = 1e-3

JPEG_QUALITY = 95
BORDER_FILL = 114  # màu xám chuẩn YOLO/Ultralytics cho vùng viền trống
BATCH_SIZE = 64  # số job lập kế hoạch mỗi vòng trước khi chạy song song

# Bảng màu BGR cố định cho 10 lớp (dùng khi vẽ ảnh preview).
CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (211, 188, 0), (255, 115, 100),
)


# ---------------------------------------------------------------------------
# 1. Pipeline albumentations (tương thích nhiều phiên bản)
# ---------------------------------------------------------------------------
def _build(transform_cls: type, variants: Sequence[dict[str, Any]] = (),
           **common: Any) -> Any:
    """Khởi tạo một transform/BboxParams bền vững giữa các phiên bản albumentations.

    Mỗi phiên bản đặt tên tham số khác nhau (ví dụ Affine: ``fill`` ở 2.x nhưng
    ``cval`` ở 1.4.x). ``variants`` là danh sách các bộ tham số thay thế, xếp
    theo thứ tự ưu tiên: chọn bộ ĐẦU TIÊN mà mọi khóa đều có trong chữ ký hàm
    khởi tạo. ``common`` là tham số chung; khóa nào không được hỗ trợ sẽ bị lọc
    bỏ (thay vì gây lỗi ``TypeError``).
    """
    params = inspect.signature(transform_cls.__init__).parameters
    chosen: dict[str, Any] = {}
    for variant in variants:
        if all(key in params for key in variant):
            chosen = variant
            break
    kwargs = {k: v for k, v in {**common, **chosen}.items() if k in params}
    return transform_cls(**kwargs)


def get_train_transform(min_visibility: float = 0.4,
                        min_area: float = 4.0) -> A.Compose:
    """Tạo pipeline tăng cường mô phỏng điều kiện thực tế của camera trên xe RC.

    Thứ tự phép biến đổi bám theo "đường đi của ánh sáng" vào camera:
    hình học cảnh (Affine) -> ánh sáng sa bàn (màu/độ sáng) -> chuyển động khi
    phơi sáng (Motion Blur) -> nhiễu cảm biến -> nén JPEG khi truyền ảnh.

    KHÔNG dùng HorizontalFlip / VerticalFlip vì lật ảnh làm SAI NGỮ NGHĨA nhãn:
      * Biển "One-way road" (4) có mũi tên chỉ hướng - lật ngang thì mũi tên
        chỉ ngược chiều nhưng nhãn vẫn giữ nguyên -> dạy mô hình điều sai.
      * Biển "Roundabout" (7) có các mũi tên xoay theo một chiều nhất định
        (ngược chiều kim đồng hồ ở nước đi bên phải); ảnh lật là chiều ngược lại.
      * "Highway entrance" (1) / "Highway exit" (2) có hình vẽ làn đường nhập /
        tách về một phía; lật ngang khiến hai lớp dễ bị nhầm lẫn với nhau.
      * Chữ "STOP", ký hiệu "P" (Parking) bị lật thành chữ ngược.
      * Lật dọc tạo ra thế giới "lộn ngược" mà camera gắn cố định trên xe không
        bao giờ thấy được.
    Tóm lại: biển báo "phiên bản gương" không tồn tại ngoài thực tế.

    LƯU Ý khi train bằng Ultralytics: mặc định ``fliplr=0.5`` sẽ lật ngang
    ONLINE một nửa số ảnh -> PHẢI đặt ``fliplr=0.0`` (và ``flipud=0.0``) trong
    cấu hình train, nếu không mọi lý do ở trên đều bị vô hiệu. Tương tự,
    ``hsv_h`` của Ultralytics CỘNG DỒN với phép dịch hue ở đây nên giữ nhỏ.

    Mosaic cũng KHÔNG nằm ở đây: Ultralytics trainer áp dụng Mosaic ONLINE
    (``mosaic=1.0`` trong cấu hình train), mỗi epoch ghép 4 ảnh ngẫu nhiên khác
    nhau, đa dạng hơn hẳn so với lưu cố định ra đĩa.

    Args:
        min_visibility: Tỷ lệ diện tích tối thiểu của box còn lại trong ảnh sau
            biến đổi (so với box ban đầu). Box bị cắt quá nhiều sẽ bị loại để
            tránh nhãn "một mẩu biển báo không nhận ra được".
        min_area: Diện tích tối thiểu (pixel) của box sau biến đổi. Đặt rất nhỏ
            để không vô tình xóa các biển báo ở xa (vốn đã nhỏ trong ảnh gốc).

    Returns:
        ``A.Compose`` nhận ``image``, ``bboxes`` (YOLO chuẩn hóa) và
        ``class_labels``.
    """
    transforms = [
        # --- Random Affine: xe ôm cua, nảy xóc, tiến lại gần / ra xa biển ---
        # * scale 0.8-1.2: khoảng cách từ xe tới biển thay đổi liên tục.
        #   keep_ratio=True: phóng to/thu nhỏ ĐỀU hai trục (xa/gần thì cả
        #   chiều rộng lẫn chiều cao cùng đổi) - không kéo biển tròn thành
        #   hình elip; độ méo nhẹ (nếu cần) đã có shear đảm nhận.
        # * translate ±8%: biển xuất hiện ở nhiều vị trí trong khung hình.
        # * rotate ±7°: thân xe nghiêng (roll) khi vào cua / đi qua gờ.
        # * shear ±4°: méo do rolling-shutter của camera giá rẻ khi xe lắc.
        # Viền trống tô màu 114 (xám) như Ultralytics. rotate_method="ellipse"
        # giữ box sau xoay ôm sát vật thể hơn "largest_box".
        _build(
            A.Affine,
            variants=[
                {"fill": BORDER_FILL, "border_mode": cv2.BORDER_CONSTANT},
                {"cval": BORDER_FILL, "mode": cv2.BORDER_CONSTANT},
            ],
            scale=(0.8, 1.2),
            keep_ratio=True,
            translate_percent=(-0.08, 0.08),
            rotate=(-7, 7),
            shear=(-4, 4),
            rotate_method="ellipse",
            p=0.8,
        ),
        # --- Color Jitter / HSV shift: ánh sáng sa bàn thay đổi ---
        # Đèn trong phòng, ánh nắng qua cửa sổ, bóng đổ của người đứng cạnh sa
        # bàn làm thay đổi độ sáng, độ bão hòa. HUE chỉ được dịch RẤT NHỎ: màu
        # sắc là đặc trưng phân biệt lớp (biển cấm / Stop màu ĐỎ, biển chỉ dẫn
        # / Parking / One-way màu XANH DƯƠNG); dịch hue lớn có thể biến đỏ
        # thành tím/xanh và làm hỏng chính đặc trưng đó.
        A.OneOf(
            [
                _build(A.HueSaturationValue, hue_shift_limit=4,
                       sat_shift_limit=25, val_shift_limit=25, p=1.0),
                _build(A.ColorJitter, brightness=0.3, contrast=0.3,
                       saturation=0.3, hue=0.015, p=1.0),
            ],
            p=0.8,
        ),
        # --- Độ sáng / tương phản / gamma: auto-exposure của camera ---
        # Camera tự chỉnh phơi sáng khi xe đi từ vùng tối sang vùng sáng, ảnh
        # có thể bị cháy sáng hoặc thiếu sáng trong vài khung hình.
        A.OneOf(
            [
                _build(A.RandomBrightnessContrast, brightness_limit=0.25,
                       contrast_limit=0.25, p=1.0),
                _build(A.RandomGamma, gamma_limit=(70, 140), p=1.0),
            ],
            p=0.5,
        ),
        # --- Motion Blur: camera rung lắc khi xe chạy nhanh ---
        # Thời gian phơi sáng của camera giá rẻ dài, xe chạy nhanh hoặc rung
        # trên mặt sa bàn gồ ghề làm ảnh nhòe theo một hướng. Kernel 3-9 px
        # tương ứng mức nhòe từ nhẹ đến khá mạnh; hướng nhòe ngẫu nhiên vì rung
        # lắc xảy ra theo mọi phương.
        _build(A.MotionBlur, blur_limit=(3, 9), p=0.35),
        # --- Nhiễu cảm biến: ánh sáng yếu -> ISO cao -> ảnh "hạt" ---
        A.OneOf(
            [
                _build(A.GaussNoise,
                       variants=[{"std_range": (0.02, 0.06)},
                                 {"var_limit": (10.0, 50.0)}],
                       p=1.0),
                _build(A.ISONoise, color_shift=(0.01, 0.04),
                       intensity=(0.1, 0.4), p=1.0),
            ],
            p=0.25,
        ),
        # --- Nén JPEG: ảnh truyền qua Wi-Fi / MJPEG stream bị mất chi tiết ---
        _build(
            A.ImageCompression,
            variants=[{"quality_range": (60, 95)},
                      {"quality_lower": 60, "quality_upper": 95}],
            p=0.3,
        ),
    ]
    # BboxParams: format YOLO (cx, cy, w, h chuẩn hóa); label_fields giữ class
    # id đi kèm từng box (id được giữ NGUYÊN, albumentations không đụng vào).
    # clip=True cắt box về trong ảnh thay vì ném lỗi khi lệch số thực nhỏ.
    bbox_params = _build(
        A.BboxParams,
        variants=[{"clip": True, "filter_invalid_bboxes": True},
                  {"clip": True}],
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=min_visibility,
        min_area=min_area,
        min_width=1.0,
        min_height=1.0,
    )
    return A.Compose(transforms, bbox_params=bbox_params)


def seed_transform(transform: Any, seed: int) -> None:
    """Gieo seed cho pipeline để mỗi ảnh sinh ra có thể tái lập chính xác.

    albumentations 2.x: ``set_random_seed`` của Compose truyền CÙNG một seed
    xuống mọi transform con -> các transform rút ra cùng một dãy số ngẫu nhiên
    (ví dụ quyết định "có áp dụng không" bị tương quan). Vì vậy ta gán cho mỗi
    transform con một seed RIÊNG sinh từ ``np.random.SeedSequence``.

    albumentations 1.x dùng RNG toàn cục của ``random``/``numpy`` -> gieo seed
    toàn cục (khi đó phải chạy tuần tự 1 luồng để đảm bảo tất định).
    """
    if not hasattr(transform, "set_random_seed"):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        return
    _seed_recursively(transform, np.random.SeedSequence(seed))


def _seed_recursively(transform: Any, seed_seq: np.random.SeedSequence) -> None:
    """Gán seed riêng biệt cho transform và (đệ quy) cho từng transform con."""
    transform.set_random_seed(int(seed_seq.generate_state(1)[0]))
    if not isinstance(transform, BaseCompose):
        return
    children = list(transform.transforms)
    for child, child_seq in zip(children, seed_seq.spawn(len(children))):
        _seed_recursively(child, child_seq)


# ---------------------------------------------------------------------------
# 2. Tiện ích đọc/ghi ảnh & nhãn YOLO
# ---------------------------------------------------------------------------
def natural_sort_key(text: str) -> list[Any]:
    """Khóa sắp xếp tự nhiên: "frame_2" đứng trước "frame_10"."""
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", text)]


def imread_unicode(path: Path) -> np.ndarray | None:
    """Đọc ảnh an toàn với đường dẫn Unicode trên Windows (None nếu hỏng)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray,
                    jpeg_quality: int = JPEG_QUALITY) -> bool:
    """Ghi ảnh an toàn với đường dẫn Unicode (encode bằng OpenCV rồi tofile).

    Trả về False (thay vì ném lỗi) nếu phần mở rộng không phải định dạng ảnh
    OpenCV hỗ trợ hoặc không ghi được file.
    """
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    try:
        ok, buffer = cv2.imencode(path.suffix or ".jpg", image, params)
        if ok:
            buffer.tofile(str(path))
    except (cv2.error, OSError):
        return False
    return bool(ok)


def sanitize_yolo_boxes(boxes: np.ndarray, classes: np.ndarray,
                        eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Kẹp box YOLO về [0, 1] và loại box suy biến.

    Nhãn Roboflow thỉnh thoảng tràn số thực nhỏ (ví dụ x_max = 1.0000001);
    albumentations sẽ ném ``ValueError`` nếu tọa độ vượt [0, 1]. Ta đổi sang
    (x1, y1, x2, y2), kẹp về [0, 1] rồi đổi lại (cx, cy, w, h).
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float64), np.zeros(0, np.int64)
    cx, cy, w, h = boxes.T
    x1 = np.clip(cx - w / 2, 0.0, 1.0)
    y1 = np.clip(cy - h / 2, 0.0, 1.0)
    x2 = np.clip(cx + w / 2, 0.0, 1.0)
    y2 = np.clip(cy + h / 2, 0.0, 1.0)
    keep = ((x2 - x1) > eps) & ((y2 - y1) > eps)
    fixed = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
    return fixed[keep], classes[keep]


def parse_yolo_line(line: str) -> tuple[int, list[float]] | None:
    """Phân tích CHẶT một dòng nhãn YOLO; trả về None nếu dòng không hợp lệ.

    Hợp lệ khi: đúng 5 trường (dòng polygon/segmentation bị loại thay vì bị
    hiểu nhầm thành box); class id là số nguyên (chấp nhận "3.0" nhưng KHÔNG
    chấp nhận "6.7" - làm tròn sẽ âm thầm đổi id, trái quy ước giữ nguyên id);
    mọi tọa độ hữu hạn, tâm trong [0, 1] và w, h trong (0, 1] (có dung sai
    nhỏ cho lỗi làm tròn của Roboflow).
    """
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cls_value = float(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    # Chặn cả id khổng lồ ("1e20") vì không vừa kiểu int64 của numpy.
    if not (np.isfinite(cls_value) and cls_value.is_integer()
            and 0 <= cls_value < 2**31 and np.all(np.isfinite(coords))):
        return None
    cx, cy, w, h = coords
    lo, hi = -COORD_TOLERANCE, 1.0 + COORD_TOLERANCE
    if not (lo <= cx <= hi and lo <= cy <= hi and 0 < w <= hi and 0 < h <= hi):
        return None
    return int(cls_value), coords


def read_yolo_labels(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Đọc file nhãn YOLO -> (classes[N], boxes[N, 4]). File thiếu = ảnh negative.

    Dòng sai định dạng bị bỏ qua (xem ``parse_yolo_line``). Đọc bằng
    ``utf-8-sig`` để BOM đầu file không làm mất box đầu tiên.

    Raises:
        ValueError: file không đọc/giải mã được (ví dụ UTF-16 do PowerShell 5.1
            ghi ra) - phía gọi quyết định bỏ qua ảnh đó thay vì dừng cả lần chạy.
    """
    classes: list[int] = []
    boxes: list[list[float]] = []
    if not label_path.is_file():
        return np.zeros(0, np.int64), np.zeros((0, 4), np.float64)
    try:
        text = label_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Không đọc được nhãn {label_path.name}: {exc}") from exc
    for line in text.splitlines():
        parsed = parse_yolo_line(line)
        if parsed is not None:
            classes.append(parsed[0])
            boxes.append(parsed[1])
    return (np.asarray(classes, dtype=np.int64),
            np.asarray(boxes, dtype=np.float64).reshape(-1, 4))


def load_source_labels(label_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Nhãn dùng để tăng cường: chỉ lớp 0..9 (id GIỮ NGUYÊN) và box đã kẹp.

    Đây là nguồn DUY NHẤT cho cả việc đếm box (chỉ mục Train) lẫn việc sinh
    ảnh, nên số đếm "before" khớp đúng với những box thực sự được tăng cường.

    Returns:
        (classes[N], boxes[N, 4], số box bị bỏ vì id ngoài 0..9).
    """
    classes, boxes = read_yolo_labels(label_path)
    keep = np.isin(classes, KEEP_CLASS_IDS)  # giữ id gốc 0..9, không remap
    boxes, classes = sanitize_yolo_boxes(boxes[keep], classes[keep])
    return classes, boxes, int((~keep).sum())


def write_yolo_labels(label_path: Path, boxes: np.ndarray,
                      classes: np.ndarray) -> None:
    """Ghi nhãn YOLO dạng "%d %.6f %.6f %.6f %.6f" (box đã được kẹp về [0, 1]).

    ``newline="\\n"``: luôn xuống dòng kiểu LF như ``clean_and_filter.py`` (trên
    Windows mặc định là CRLF -> thư mục labels/ bị lẫn hai kiểu xuống dòng).
    """
    boxes, classes = sanitize_yolo_boxes(np.asarray(boxes, np.float64),
                                         np.asarray(classes, np.int64))
    lines = [f"{int(c)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
             for c, b in zip(classes, boxes)]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                          encoding="utf-8", newline="\n")


def augment_sample(image: np.ndarray, bboxes: np.ndarray,
                   class_labels: np.ndarray, transform: A.Compose,
                   required_class: int | None = None,
                   max_retries: int = 10,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Tăng cường một ảnh, thử lại nếu kết quả không đạt.

    Một lần thử bị coi là KHÔNG đạt nếu box của ``required_class`` (lớp thiểu
    số đang cần bổ sung) bị loại do ``min_visibility`` - ảnh như vậy không giúp
    cân bằng lớp. Trong các lần đạt, ưu tiên lần giữ được NHIỀU box nhất: mỗi
    box bị rơi mất mà vật thể vẫn còn thấy một phần trong ảnh chính là một
    "vật thể không có nhãn", dạy mô hình coi biển báo là nền. Giữ đủ box thì
    trả về ngay.

    Returns:
        (ảnh, boxes[N, 4], classes[N]) hoặc None nếu mọi lần thử đều thất bại.
    """
    boxes, classes = sanitize_yolo_boxes(np.asarray(bboxes, np.float64),
                                         np.asarray(class_labels, np.int64))
    if len(boxes) == 0:
        return None
    best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for _ in range(max_retries):
        out = transform(image=image, bboxes=boxes,
                        class_labels=classes.tolist())
        out_boxes = np.asarray(out["bboxes"], np.float64).reshape(-1, 4)
        out_classes = np.asarray(out["class_labels"], np.int64)
        if required_class is not None and required_class not in out_classes:
            continue
        if best is None or len(out_classes) > len(best[2]):
            best = (out["image"], out_boxes, out_classes)
        if len(out_classes) == len(classes):
            break
    return best


# ---------------------------------------------------------------------------
# 3. Quét tập Train
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainIndex:
    """Chỉ mục các ảnh GỐC trong Train (không tính file _aug)."""

    image_paths: list[Path]
    label_paths: list[Path]
    class_matrix: np.ndarray  # [num_images, NUM_CLASSES]: số box mỗi lớp/ảnh
    group_ids: np.ndarray  # [num_images]: chỉ số nhóm "frame gốc" của ảnh
    ignored_boxes: int  # box có id ngoài 0..9 (không nên có sau bước clean)
    unreadable_labels: list[str]  # file nhãn không giải mã được (bị bỏ qua)

    @property
    def num_negatives(self) -> int:
        """Số ảnh negative (nhãn rỗng), không tính ảnh có nhãn đọc lỗi."""
        empty = int((self.class_matrix.sum(axis=1) == 0).sum())
        return empty - len(self.unreadable_labels)


def _ensure_train_dir(train_dir: Path, force: bool) -> None:
    """Chặn việc tăng cường nhầm vào Val/Test.

    Val/Test phải phản ánh đúng phân phối dữ liệu THẬT mà xe sẽ gặp. Nếu sinh
    ảnh tăng cường vào đó: (1) chỉ số mAP bị bóp méo vì đánh giá trên ảnh nhân
    tạo; (2) ảnh _aug sinh từ cùng một khung hình gốc với ảnh train là
    "anh em sinh đôi" -> rò rỉ dữ liệu (data leakage), mô hình được kiểm tra
    trên thứ gần như đã thấy. Vì vậy chỉ cho phép thư mục Train, trừ khi người
    dùng chủ động truyền ``--force``.
    """
    if not (train_dir / "images").is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy {train_dir / 'images'} - hãy chạy "
            "clean_and_filter.py trước hoặc chỉ định --train-dir.")
    if train_dir.name.lower() in NON_TRAIN_DIR_NAMES and not force:
        raise ValueError(
            f"'{train_dir}' trông giống tập Val/Test. Chỉ được tăng cường "
            "tập Train (tránh data leakage). Dùng --force nếu thực sự muốn.")


def _list_images(train_dir: Path) -> list[Path]:
    images_dir = train_dir / "images"
    if not images_dir.is_dir():
        return []
    return [p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def find_generated_files(train_dir: Path) -> list[Path]:
    """Tìm ảnh/nhãn do script này sinh ra ở lần chạy trước.

    Một file được coi là "đã sinh" khi tên có đuôi ``_augNNN`` VÀ ảnh "cha"
    (tên bỏ đuôi đó) đang có trong ``images/``. Điều kiện thứ hai bảo vệ ảnh
    GỐC tình cờ đặt tên kiểu ``myclip_aug001.jpg`` khỏi bị xóa nhầm. So sánh
    không phân biệt hoa/thường vì hệ thống file Windows cũng vậy.
    """
    images = _list_images(train_dir)
    stems = {p.stem.casefold() for p in images}

    def is_generated(path: Path) -> bool:
        match = AUG_STEM_RE.search(path.stem)
        return bool(match) and path.stem[:match.start()].casefold() in stems

    labels_dir = train_dir / "labels"
    labels = [p for p in labels_dir.iterdir()
              if p.is_file() and p.suffix.lower() == ".txt"] \
        if labels_dir.is_dir() else []
    return [p for p in images + labels if is_generated(p)]


def remove_generated_files(train_dir: Path) -> int:
    """Xóa mọi ảnh/nhãn ``*_augNNN`` của lần chạy trước (đảm bảo idempotent)."""
    generated = find_generated_files(train_dir)
    for path in generated:
        path.unlink()
    return len(generated)


def list_original_images(train_dir: Path) -> list[Path]:
    """Liệt kê ảnh gốc (bỏ file _aug), sắp xếp tự nhiên để kết quả tất định."""
    generated = set(find_generated_files(train_dir))
    images = [p for p in _list_images(train_dir) if p not in generated]
    return sorted(images, key=lambda p: natural_sort_key(p.name))


# Bản sao quy tắc ``source_key_from_name`` của clean_and_filter.py (mỗi script
# tự chứa, không import lẫn nhau) - PHẢI giữ hai bản giống nhau.
_RF_SUFFIX_RE = re.compile(r"\.rf\.[0-9a-f]{8,}$", re.IGNORECASE)
_EXT_TOKEN_RE = re.compile(r"_(jpe?g|png|bmp|webp)$", re.IGNORECASE)
_COPY_SUFFIX_RE = re.compile(
    r"(?:[\s_-]+(?:kopya|copy|kopie|copie)(?:\s*\(\d+\))?|\s*\(\d+\))$",
    re.IGNORECASE)


def source_key_from_name(filename: str) -> str:
    """Suy ra khóa FRAME GỐC từ tên file Roboflow.

    ``frame_0492_jpg.rf.<hash>.jpg`` -> ``frame_0492_jpg``;
    ``yaya3_mp4-12-Kopya_jpg.rf.<hash>.jpg`` -> ``yaya3_mp4-12_jpg``.
    Roboflow đã nhân bản mỗi frame tới 21 lần (mỗi bản một hậu tố ``.rf.``);
    gom chúng về một khóa để giới hạn ``max_copies`` áp dụng cho cả FRAME, không
    phải cho từng file - tránh một cảnh duy nhất bị nhân bản quá nhiều lần.
    """
    stem = _RF_SUFFIX_RE.sub("", Path(filename).stem)
    match = _EXT_TOKEN_RE.search(stem)
    ext_token = match.group(1).lower() if match else ""
    if match:
        stem = stem[: match.start()]
    while True:  # có thể có nhiều hậu tố copy chồng nhau
        stripped = _COPY_SUFFIX_RE.sub("", stem).rstrip(" _-")
        if stripped == stem or not stripped:
            break
        stem = stripped
    stem = stem.lower()
    return f"{stem}_{ext_token}" if ext_token else stem


def _safe_load_labels(label_path: Path
                      ) -> tuple[np.ndarray, np.ndarray, int] | ValueError:
    """``load_source_labels`` nhưng trả lỗi về (thay vì ném) để chạy trong pool."""
    try:
        return load_source_labels(label_path)
    except ValueError as exc:
        return exc


def build_train_index(train_dir: Path, workers: int) -> TrainIndex:
    """Đọc toàn bộ nhãn (song song) và lập ma trận số box theo lớp cho mỗi ảnh.

    Số đếm lấy từ ``load_source_labels`` (đã lọc lớp + kiểm tra hợp lệ), nên
    dòng hỏng / box suy biến không bị tính là "instance". File nhãn không giải
    mã được được cảnh báo và ảnh đó không bao giờ được dùng làm nguồn.
    """
    image_paths = list_original_images(train_dir)
    label_paths = [train_dir / "labels" / f"{p.stem}.txt" for p in image_paths]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_safe_load_labels, label_paths))
    matrix = np.zeros((len(image_paths), NUM_CLASSES), dtype=np.int64)
    ignored = 0
    unreadable: list[str] = []
    for row, result in enumerate(results):
        if isinstance(result, ValueError):
            unreadable.append(label_paths[row].name)
            print(f"[aug] CẢNH BÁO: {result} -> bỏ qua ảnh này.")
            continue
        classes, _, num_ignored = result
        ignored += num_ignored
        matrix[row] = np.bincount(classes, minlength=NUM_CLASSES)
    key_to_group: dict[str, int] = {}
    group_ids = np.array(
        [key_to_group.setdefault(source_key_from_name(p.name), len(key_to_group))
         for p in image_paths], dtype=np.int64)
    return TrainIndex(image_paths, label_paths, matrix, group_ids, ignored,
                      unreadable)


def load_class_names(train_dir: Path) -> dict[int, str]:
    """Đọc tên lớp từ dataset_filtered.yaml (cạnh thư mục Train), có dự phòng."""
    names = dict(DEFAULT_CLASS_NAMES)
    yaml_path = train_dir.parent / "dataset_filtered.yaml"
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return names
    raw = data.get("names")
    items = raw.items() if isinstance(raw, dict) else enumerate(raw or [])
    for key, value in items:
        if int(key) in names:
            names[int(key)] = str(value)
    return names


# ---------------------------------------------------------------------------
# 4. Cân bằng lớp (Class Balancing)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AugJob:
    """Một yêu cầu sinh ảnh: ảnh nguồn + lớp cần bổ sung + seed riêng."""

    source_index: int
    image_path: Path
    label_path: Path
    out_stem: str
    required_class: int
    seed: int


def resolve_target(counts: np.ndarray, target: str | int = "max",
                   target_ratio: float | None = None) -> int:
    """Xác định số box mục tiêu cho mỗi lớp.

    * ``target="max"``: bằng lớp đông nhất (mặc định).
    * ``target=<int>``: một con số cố định.
    * ``target_ratio``: tỷ lệ so với lớp đông nhất (ưu tiên hơn ``target``).
    """
    max_count = int(counts.max()) if counts.size else 0
    if target_ratio is not None:
        return int(round(max_count * target_ratio))
    if str(target).lower() == "max":
        return max_count
    return int(target)


def absent_classes(counts: np.ndarray) -> list[int]:
    """Các lớp 0..9 không có box nào (không thể cân bằng vì không có ảnh nguồn)."""
    return [c for c in KEEP_CLASS_IDS if counts[c] == 0]


def imbalance_ratio(counts: np.ndarray) -> float | None:
    """Tỷ lệ mất cân bằng = max / min trên CẢ 10 lớp.

    Trả về None (JSON ``null``, nghĩa là "vô cùng") nếu có lớp nào 0 box: bỏ
    qua lớp vắng mặt sẽ cho ra con số "gần cân bằng" gây hiểu nhầm trong báo
    cáo. Danh sách lớp vắng mặt được ghi riêng (``absent_classes_*``).
    """
    if counts.size == 0 or counts.min() == 0:
        return None
    return round(float(counts.max() / counts.min()), 4)


@dataclass
class _PlanState:
    """Trạng thái lập kế hoạch dùng chung giữa các batch."""

    group_copies: np.ndarray  # số ảnh đã sinh cho mỗi nhóm frame gốc
    blocked: np.ndarray  # ảnh nguồn đã thất bại -> không chọn lại
    used_stems: set[str]  # stem (casefold) đã có trên đĩa / đã lên kế hoạch
    exhausted: set[int]  # lớp đã hết ảnh nguồn hợp lệ

    def effective_copies(self, group_ids: np.ndarray,
                         max_copies: int) -> np.ndarray:
        """Số lần nhân bản của mỗi ẢNH = số lần của NHÓM frame gốc chứa nó."""
        copies = self.group_copies[group_ids].copy()
        copies[self.blocked] = max_copies
        return copies

    def next_out_stem(self, stem: str) -> str:
        """Tên ``<stem>_augNNN`` chưa dùng (so sánh không phân biệt hoa/thường).

        Windows coi ``Sign`` và ``sign`` là một tên: nếu đánh số riêng theo
        từng stem, ``Sign.jpg`` và ``sign.png`` sẽ ghi đè ``_aug`` của nhau.
        """
        number = 1
        while f"{stem}{AUG_TAG}{number:03d}".casefold() in self.used_stems:
            number += 1
        out_stem = f"{stem}{AUG_TAG}{number:03d}"
        self.used_stems.add(out_stem.casefold())
        return out_stem


def select_source(class_matrix: np.ndarray, deficit: np.ndarray, target: int,
                  copies: np.ndarray, required_class: int, max_copies: int,
                  rng: np.random.Generator, majority_penalty: float = 1.0,
                  top_k: int = 8) -> int | None:
    """Chọn ảnh nguồn tốt nhất để bổ sung ``required_class``.

    Điểm của một ảnh (n_c = số box lớp c trong ảnh, d_c = deficit lớp c):

        score = Σ n_c · max(d_c, 0) / target  -  λ · Σ n_c · [d_c <= 0]

    -> ưu tiên ảnh chứa NHIỀU box thuộc các lớp còn thiếu và phạt ảnh chứa box
    của lớp ĐÃ ĐỦ (nhân bản chúng chỉ làm lớp đa số đông thêm). Lấy nhóm
    ``top_k`` điểm cao nhất (kể cả các ảnh hòa điểm), trong nhóm đó ưu tiên ảnh
    đã được nhân bản ÍT lần nhất (trải đều cho đa dạng), rồi chọn ngẫu nhiên có
    seed. ``copies`` được tính theo NHÓM frame gốc (mọi bản sao Roboflow của
    cùng một frame dùng chung một bộ đếm), mỗi frame nhân bản tối đa
    ``max_copies`` lần để tránh overfit vào vài cảnh duy nhất.
    """
    candidates = np.flatnonzero((class_matrix[:, required_class] > 0)
                                & (copies < max_copies))
    if candidates.size == 0:
        return None
    sub = class_matrix[candidates].astype(np.float64)
    gain = sub @ (np.clip(deficit, 0, None) / max(target, 1))
    penalty = sub @ (deficit <= 0).astype(np.float64)
    scores = gain - majority_penalty * penalty
    kth = np.sort(scores)[::-1][min(top_k, scores.size) - 1]
    pool = candidates[scores >= kth - 1e-9]
    pool = pool[copies[pool] == copies[pool].min()]
    return int(rng.choice(pool))


def _plan_batch(index: TrainIndex, counts: np.ndarray, target: int,
                state: _PlanState, max_copies: int, limit: int,
                rng: np.random.Generator) -> list[AugJob]:
    """Lập kế hoạch tối đa ``limit`` job theo chiến lược tham lam (greedy).

    Trong batch, số box được cập nhật theo KỲ VỌNG (giả sử mọi box của ảnh
    nguồn đều sống sót); sau khi chạy xong sẽ hiệu chỉnh bằng số box THỰC TẾ.
    Ở mỗi bước, lớp được chọn là lớp có deficit tương đối (deficit / target)
    lớn nhất - tức lớp đang thiếu trầm trọng nhất.
    """
    expected = counts.astype(np.float64).copy()
    jobs: list[AugJob] = []
    while len(jobs) < limit:
        deficit = target - expected
        open_classes = [c for c in KEEP_CLASS_IDS
                        if deficit[c] > 0 and c not in state.exhausted]
        if not open_classes:
            break
        cls = max(open_classes, key=lambda c: (deficit[c] / target, -c))
        copies = state.effective_copies(index.group_ids, max_copies)
        src = select_source(index.class_matrix, deficit, target, copies, cls,
                            max_copies, rng)
        if src is None:  # lớp này hết ảnh nguồn hợp lệ (đã chạm max_copies)
            state.exhausted.add(cls)
            continue
        state.group_copies[index.group_ids[src]] += 1
        expected += index.class_matrix[src]
        jobs.append(AugJob(
            source_index=src,
            image_path=index.image_paths[src],
            label_path=index.label_paths[src],
            out_stem=state.next_out_stem(index.image_paths[src].stem),
            required_class=cls,
            seed=int(rng.integers(0, 2**31 - 1)),
        ))
    return jobs


class _AugWorker:
    """Thực thi job trong thread pool; mỗi luồng giữ một pipeline riêng."""

    def __init__(self, train_dir: Path, transform_kwargs: dict[str, Any],
                 max_retries: int) -> None:
        self.images_dir = train_dir / "images"
        self.labels_dir = train_dir / "labels"
        self.transform_kwargs = transform_kwargs
        self.max_retries = max_retries
        self._local = threading.local()

    def _transform(self) -> A.Compose:
        if not hasattr(self._local, "transform"):
            self._local.transform = get_train_transform(**self.transform_kwargs)
        return self._local.transform

    def __call__(self, job: AugJob) -> np.ndarray | None:
        """Sinh 1 ảnh + nhãn; trả về số box mỗi lớp đã ghi (None nếu thất bại)."""
        image = imread_unicode(job.image_path)
        if image is None:
            return None
        try:
            classes, boxes, _ = load_source_labels(job.label_path)
        except ValueError:  # file nhãn bị thay đổi/hỏng giữa chừng
            return None
        transform = self._transform()
        seed_transform(transform, job.seed)
        result = augment_sample(image, boxes, classes, transform,
                                required_class=job.required_class,
                                max_retries=self.max_retries)
        if result is None:
            return None
        aug_image, aug_boxes, aug_classes = result
        aug_boxes, aug_classes = sanitize_yolo_boxes(aug_boxes, aug_classes)
        if not imwrite_unicode(self.images_dir / f"{job.out_stem}.jpg",
                               aug_image):
            return None
        write_yolo_labels(self.labels_dir / f"{job.out_stem}.txt",
                          aug_boxes, aug_classes)
        return np.bincount(aug_classes, minlength=NUM_CLASSES)[:NUM_CLASSES]


def apply_augmentation(train_dir: str | Path = DEFAULT_TRAIN_DIR,
                       target: str | int = "max",
                       target_ratio: float | None = None,
                       max_copies_per_image: int = 5,
                       max_new_images: int | None = None,
                       seed: int = 42,
                       min_visibility: float = 0.4,
                       max_retries: int = 10,
                       workers: int | None = None,
                       report_path: str | Path | None = None,
                       force: bool = False) -> dict[str, Any]:
    """Quét tập Train và sinh thêm ảnh cho các lớp thiểu số để cân bằng dữ liệu.

    Quy trình:
      1. Đếm box mỗi lớp CHỈ trên ảnh gốc, rồi xóa file ``_augNNN`` cũ
         (idempotent). Lập chỉ mục TRƯỚC khi xóa: nếu dữ liệu đầu vào có vấn
         đề thì lỗi xảy ra khi bộ ``_aug`` cũ vẫn còn nguyên.
      2. target = max(count) | số cố định | tỷ lệ * max; deficit_c = target - count_c.
      3. Lặp tham lam: chọn lớp thiếu nhiều nhất -> chọn ảnh nguồn theo điểm
         (xem ``select_source``) -> tăng cường -> lưu ``<stem>_augNNN.jpg/.txt``
         -> cập nhật số đếm bằng TẤT CẢ box thực sự được tạo ra (kể cả box của
         lớp khác đi kèm trong ảnh). Ảnh nguồn nào thất bại (không đọc được,
         hoặc mất box cần bổ sung sau mọi lần thử) sẽ không được chọn lại.
      4. Dừng khi mọi deficit <= 0, hết ảnh nguồn, hoặc chạm ``max_new_images``.
    Ảnh negative (nhãn rỗng) không bao giờ được dùng làm nguồn.
    ``max_copies_per_image`` áp dụng cho mỗi FRAME GỐC (gom mọi bản sao
    Roboflow ``.rf.<hash>`` của cùng frame, xem ``source_key_from_name``).

    Returns:
        Báo cáo dạng dict (cũng được lưu ra ``augmentation_report.json``).
    """
    started = time.perf_counter()
    if max_copies_per_image < 1 or (workers is not None and workers < 1):
        raise ValueError("max_copies_per_image và workers phải >= 1")
    if target_ratio is not None and not (np.isfinite(target_ratio)
                                         and target_ratio > 0):
        raise ValueError("target_ratio phải là số dương hữu hạn")
    train_dir = Path(train_dir).resolve()
    _ensure_train_dir(train_dir, force)
    (train_dir / "labels").mkdir(parents=True, exist_ok=True)
    # albumentations 1.x dùng RNG toàn cục -> chỉ tất định khi chạy 1 luồng.
    if not hasattr(A.Compose, "set_random_seed"):
        workers = 1
    workers = workers or min(8, os.cpu_count() or 1)

    index = build_train_index(train_dir, workers)
    removed = remove_generated_files(train_dir)
    print(f"[aug] Đã xóa {removed} file _aug của lần chạy trước.")
    before = index.class_matrix.sum(axis=0)
    num_negatives = index.num_negatives
    goal = resolve_target(before, target, target_ratio)
    num_groups = int(index.group_ids.max()) + 1 if index.group_ids.size else 0
    print(f"[aug] {len(index.image_paths)} ảnh gốc ({num_negatives} negative, "
          f"{num_groups} frame gốc), target = {goal} box/lớp.")
    missing = absent_classes(before)
    if missing:
        print(f"[aug] CẢNH BÁO: lớp {missing} không có box nào trong Train "
              "-> không thể cân bằng các lớp này.")

    rng = np.random.default_rng(seed)
    counts = before.copy()
    state = _PlanState(
        group_copies=np.zeros(num_groups, dtype=np.int64),
        blocked=np.zeros(len(index.image_paths), dtype=bool),
        used_stems={p.stem.casefold() for p in _list_images(train_dir)},
        exhausted=set(),
    )
    generated = np.zeros(NUM_CLASSES, dtype=np.int64)
    num_generated = num_failed = 0
    worker = _AugWorker(train_dir, {"min_visibility": min_visibility},
                        max_retries)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            budget = BATCH_SIZE if max_new_images is None else min(
                BATCH_SIZE, max_new_images - num_generated)
            if budget <= 0:
                stop_reason = "reached max_new_images"
                break
            jobs = _plan_batch(index, counts, goal, state,
                               max_copies_per_image, budget, rng)
            if not jobs:
                stop_reason = ("all deficits filled" if not state.exhausted
                               else "source images exhausted for some classes")
                break
            # pool.map giữ đúng thứ tự job -> cập nhật số đếm tất định.
            for job, produced in zip(jobs, pool.map(worker, jobs)):
                if produced is None:
                    num_failed += 1
                    state.blocked[job.source_index] = True
                    continue
                counts += produced
                generated += produced
                num_generated += 1
            remaining = int(np.clip(goal - counts, 0, None).sum())
            print(f"[aug] sinh {num_generated} ảnh | box còn thiếu: "
                  f"{remaining}")

    names = load_class_names(train_dir)
    report: dict[str, Any] = {
        "train_dir": train_dir.as_posix(),
        "albumentations_version": A.__version__,
        "seed": seed,
        "target_mode": ("ratio" if target_ratio is not None else str(target)),
        "target_per_class": goal,
        "max_copies_per_image": max_copies_per_image,
        "max_copies_scope": "source frame (Roboflow .rf.<hash> copies grouped)",
        "max_new_images": max_new_images,
        "min_visibility": min_visibility,
        "removed_previous_aug_files": removed,
        "num_original_images": len(index.image_paths),
        "num_source_frames": num_groups,
        "num_negative_images": num_negatives,
        "num_generated_images": num_generated,
        "num_failed_jobs": num_failed,
        "ignored_boxes_outside_0_9": index.ignored_boxes,
        "unreadable_label_files": index.unreadable_labels,
        "stop_reason": stop_reason,
        "exhausted_classes": sorted(state.exhausted),
        "absent_classes_before": absent_classes(before),
        "absent_classes_after": absent_classes(counts),
        "imbalance_ratio_before": imbalance_ratio(before),
        "imbalance_ratio_after": imbalance_ratio(counts),
        "recommended_train_hyp": RECOMMENDED_TRAIN_HYP,
        "per_class": [
            {"id": c, "name": names[c], "before": int(before[c]),
             "generated": int(generated[c]), "after": int(counts[c]),
             "remaining_deficit": int(max(goal - counts[c], 0))}
            for c in KEEP_CLASS_IDS
        ],
        "total_boxes_before": int(before.sum()),
        "total_boxes_after": int(counts.sum()),
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }
    report_file = (Path(report_path) if report_path
                   else train_dir.parent / "augmentation_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print_report(report)
    print(f"[aug] Báo cáo: {report_file}")
    return report


def print_report(report: dict[str, Any]) -> None:
    """In bảng thống kê trước/sau tăng cường (dễ chép vào báo cáo LaTeX)."""
    header = f"{'ID':>3} | {'Class':<22} | {'Before':>7} | {'+Aug':>6} | " \
             f"{'After':>7} | {'Missing':>7}"
    print("\n" + header + "\n" + "-" * len(header))
    for row in report["per_class"]:
        print(f"{row['id']:>3} | {row['name'][:22]:<22} | {row['before']:>7} | "
              f"{row['generated']:>6} | {row['after']:>7} | "
              f"{row['remaining_deficit']:>7}")
    print("-" * len(header))

    def ratio_text(ratio: float | None, absent: list[int]) -> str:
        return str(ratio) if ratio is not None else f"inf (lớp {absent} = 0 box)"

    ratio_before = ratio_text(report["imbalance_ratio_before"],
                              report["absent_classes_before"])
    ratio_after = ratio_text(report["imbalance_ratio_after"],
                             report["absent_classes_after"])
    print(f"Ảnh sinh thêm: {report['num_generated_images']} | "
          f"Imbalance (max/min): {ratio_before} -> {ratio_after} | "
          f"Dừng vì: {report['stop_reason']} | {report['elapsed_sec']} s")
    print("Khi train Ultralytics hãy dùng: "
          + ", ".join(f"{k}={v}" for k, v in report["recommended_train_hyp"].items())
          + "\n")


# ---------------------------------------------------------------------------
# 5. Ảnh minh họa (preview) cho báo cáo
# ---------------------------------------------------------------------------
def draw_yolo_boxes(image: np.ndarray, boxes: np.ndarray,
                    classes: np.ndarray) -> np.ndarray:
    """Vẽ box + class id lên ảnh (tọa độ YOLO chuẩn hóa)."""
    canvas = image.copy()
    h, w = canvas.shape[:2]
    thickness = max(2, round(min(h, w) / 200))
    for cls_id, (cx, cy, bw, bh) in zip(classes, boxes):
        color = CLASS_COLORS[int(cls_id) % len(CLASS_COLORS)]
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        text = str(int(cls_id))
        scale = thickness * 0.35
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                      thickness)
        ty = max(y1, th + 4)
        cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(canvas, text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Thu ảnh vào ô vuông ``size`` giữ nguyên tỷ lệ, viền màu 114."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(image, (max(1, round(w * scale)),
                                 max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    cell = np.full((size, size, 3), BORDER_FILL, dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    cell[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return cell


def _shorten(text: str, max_len: int) -> str:
    """Rút gọn tên dài ở GIỮA để vẫn thấy phần đuôi ``_augNNN``."""
    if len(text) <= max_len:
        return text
    half = (max_len - 3) // 2
    return f"{text[:half]}...{text[-half:]}"


def _preview_samples(train_dir: Path, num_samples: int, seed: int,
                     ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Lấy mẫu cho preview.

    Nếu Train đã có file _aug thì đọc chúng TỪ ĐĨA (kiểm chứng luôn nhãn đã
    ghi khớp với ảnh); nếu chưa có thì tăng cường tại chỗ (không ghi gì).
    """
    rng = np.random.default_rng(seed)
    labels_dir = train_dir / "labels"
    generated = sorted((p for p in find_generated_files(train_dir)
                        if p.suffix.lower() in IMAGE_EXTENSIONS),
                       key=lambda p: natural_sort_key(p.name))
    samples = []
    if generated:
        picks = rng.choice(len(generated), min(num_samples, len(generated)),
                           replace=False)
        for i in sorted(picks):
            path = generated[i]
            image = imread_unicode(path)
            try:
                classes, boxes = read_yolo_labels(labels_dir / f"{path.stem}.txt")
            except ValueError:
                continue
            if image is not None:
                samples.append((image, boxes, classes, path.stem))
        return samples

    transform = get_train_transform()
    originals = list_original_images(train_dir)
    order = rng.permutation(len(originals))
    for i in order:
        if len(samples) >= num_samples:
            break
        path = originals[i]
        try:
            classes, boxes, _ = load_source_labels(
                labels_dir / f"{path.stem}.txt")
        except ValueError:
            continue
        image = imread_unicode(path) if len(classes) else None
        if image is None:
            continue
        seed_transform(transform, int(rng.integers(0, 2**31 - 1)))
        result = augment_sample(image, boxes, classes, transform)
        if result is not None:
            samples.append((*result, f"{path.stem} (on-the-fly)"))
    return samples


def save_preview(train_dir: str | Path = DEFAULT_TRAIN_DIR,
                 out_path: str | Path = DEFAULT_PREVIEW_PATH,
                 num_samples: int = 16, cols: int = 4, cell_size: int = 320,
                 seed: int = 42) -> Path | None:
    """Ghép lưới ảnh tăng cường (mặc định 4x4) có vẽ box + class id.

    Dùng để kiểm tra bằng mắt rằng box vẫn ôm đúng vật thể sau Affine, và để
    chèn vào báo cáo LaTeX.
    """
    train_dir = Path(train_dir).resolve()
    samples = _preview_samples(train_dir, num_samples, seed)
    if not samples:
        print("[aug] Không có mẫu nào để tạo preview.")
        return None
    cells = []
    for image, boxes, classes, title in samples:
        # Dải tiêu đề đặt PHÍA TRÊN ảnh (không đè lên box sát mép trên).
        header = np.zeros((20, cell_size, 3), dtype=np.uint8)
        cv2.putText(header, _shorten(title, 46), (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                    cv2.LINE_AA)
        body = _letterbox(draw_yolo_boxes(image, boxes, classes), cell_size)
        cells.append(np.vstack([header, body]))
    blank = np.full_like(cells[0], BORDER_FILL)
    rows = []
    for start in range(0, len(cells), cols):
        row = cells[start:start + cols]
        row += [blank] * (cols - len(row))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(out_path, grid, jpeg_quality=92):
        print(f"[aug] LỖI: không ghi được preview {out_path} "
              "(phần mở rộng phải là định dạng ảnh, ví dụ .jpg).")
        return None
    print(f"[aug] Preview: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def _parse_target(value: str) -> str | int:
    if value.lower() == "max":
        return "max"
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--target phải là 'max' hoặc số nguyên") \
            from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("--target phải > 0")
    return number


def _int_at_least(minimum: int) -> Callable[[str], int]:
    """Tạo kiểu argparse: số nguyên >= ``minimum``."""
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"'{value}' không phải số nguyên") \
                from exc
        if number < minimum:
            raise argparse.ArgumentTypeError(f"giá trị phải >= {minimum}")
        return number
    return parse


def _float_in(low: float, high: float,
              low_inclusive: bool = True) -> Callable[[str], float]:
    """Tạo kiểu argparse: số thực hữu hạn trong [low, high] (hoặc (low, high])."""
    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"'{value}' không phải số thực") \
                from exc
        above_low = number >= low if low_inclusive else number > low
        if not (np.isfinite(number) and above_low and number <= high):
            bracket = "[" if low_inclusive else "("
            raise argparse.ArgumentTypeError(
                f"giá trị phải thuộc {bracket}{low}, {high}]")
        return number
    return parse


def _image_path(value: str) -> Path:
    """Đường dẫn preview phải có đuôi ảnh (kiểm tra TRƯỚC khi chạy tăng cường)."""
    path = Path(value)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise argparse.ArgumentTypeError(
            f"đuôi file preview phải là một trong {sorted(IMAGE_EXTENSIONS)}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tăng cường dữ liệu + cân bằng lớp (0..9) cho tập Train.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR,
                        help="Thư mục Train chứa images/ và labels/.")
    parser.add_argument("--target", type=_parse_target, default="max",
                        help="'max' (mặc định) hoặc số box mục tiêu mỗi lớp.")
    parser.add_argument("--target-ratio", type=_float_in(0.0, 100.0, False),
                        default=None,
                        help="Mục tiêu = tỷ lệ * lớp đông nhất (ghi đè --target).")
    parser.add_argument("--max-copies", type=_int_at_least(1), default=5,
                        help="Số ảnh tăng cường tối đa sinh từ MỘT frame gốc "
                             "(gom mọi bản sao Roboflow .rf.<hash>).")
    parser.add_argument("--max-new-images", type=_int_at_least(0),
                        default=None, help="Giới hạn tổng số ảnh sinh thêm.")
    parser.add_argument("--min-visibility", type=_float_in(0.0, 1.0),
                        default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=_int_at_least(1), default=None)
    parser.add_argument("--report", type=Path, default=None,
                        help="Đường dẫn JSON báo cáo "
                             "(mặc định <train-dir>/../augmentation_report.json).")
    parser.add_argument("--preview", type=_image_path,
                        default=DEFAULT_PREVIEW_PATH,
                        help="Đường dẫn ảnh preview (.jpg/.png...).")
    parser.add_argument("--no-preview", action="store_true",
                        help="Không xuất ảnh preview.")
    parser.add_argument("--preview-only", action="store_true",
                        help="Chỉ xuất preview, không sinh/xóa file nào.")
    parser.add_argument("--clean-only", action="store_true",
                        help="Chỉ xóa các file _aug đã sinh rồi thoát.")
    parser.add_argument("--force", action="store_true",
                        help="Cho phép chạy trên thư mục có tên Val/Test.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # Console Windows mặc định cp1252 không in được tiếng Việt -> ép UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    train_dir = args.train_dir.resolve()
    try:
        _ensure_train_dir(train_dir, args.force)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[aug] LỖI: {exc}", file=sys.stderr)
        return 1

    if args.clean_only:
        removed = remove_generated_files(train_dir)
        print(f"[aug] Đã xóa {removed} file _aug trong {train_dir}.")
        return 0
    if not args.preview_only:
        apply_augmentation(
            train_dir,
            target=args.target,
            target_ratio=args.target_ratio,
            max_copies_per_image=args.max_copies,
            max_new_images=args.max_new_images,
            seed=args.seed,
            min_visibility=args.min_visibility,
            workers=args.workers,
            report_path=args.report,
            force=args.force,
        )
    if args.preview_only or not args.no_preview:
        save_preview(train_dir, args.preview, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
