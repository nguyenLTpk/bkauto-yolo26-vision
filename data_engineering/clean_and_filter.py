"""
clean_and_filter.py - Giai đoạn 1 (Core Script): Làm sạch, Lọc lớp, Background
Negatives và Chia tập theo chuỗi (Sequence-based Split) cho bộ dữ liệu YOLO.

Mục đích
--------
Biến bộ dữ liệu gốc (Roboflow YOLO export, 17+ class) trong ``data/raw/`` thành
bộ dữ liệu sạch, chỉ gồm 10 class (0..9: 9 loại biển báo + 1 người đi bộ) và được
chia Train/Val/Test KHÔNG rò rỉ dữ liệu (data leakage), sẵn sàng cho Ultralytics.

Pipeline (mỗi bước đều có bộ đếm, ghi vào ``cleaning_report.json``):

    A. Quét đệ quy ảnh + ghép file nhãn theo quy ước YOLO (images/ -> labels/).
    B. Làm sạch ảnh: bỏ ảnh 0 byte, ảnh hỏng (cv2 không decode được); JPEG bị
       cắt cụt (thiếu marker EOI FFD9) mà cứu được -> sửa bằng cách encode lại.
    C. Làm sạch nhãn: bỏ dòng sai định dạng, box diện tích <= 0, tâm nằm ngoài
       ảnh, box vượt biên quá dung sai, box quá nhỏ, tỉ lệ cạnh bất thường, dòng
       trùng lặp. Box vượt biên trong dung sai (sai số làm tròn) thì được cắt (clip).
    D. Lọc lớp: GIỮ NGUYÊN class 0..9, BỎ mọi class khác (10..16, 17 ...).
       KHÔNG đánh lại chỉ số (no re-index) - đã thống nhất với giảng viên.
    E. Background negatives: xác định pool ảnh "nền" (không có đối tượng thuộc
       10 class), loại ảnh mơ hồ / đáng ngờ; chỉ giữ ngẫu nhiên 10% pool (việc
       lấy mẫu làm SAU bước F, riêng trong từng split) để tránh mất cân bằng.
    F. Sequence-based Split: gom nhóm theo frame gốc, sắp xếp alpha-numeric, cắt
       khối (chunk) liên tiếp, gán nguyên khối vào Train/Val/Test theo tỉ lệ
       80/10/10 có phân tầng (stratified) theo class, chừa "purge gap" ở biên.
    G. Xuất dữ liệu ra ``data/processed/{Train,Val,Test}/{images,labels}``, sinh
       ``dataset_filtered.yaml``, ``split_manifest.csv``, ``cleaning_report.json``
       và ``negatives_review.jpg`` (ảnh ghép để kiểm tra bằng mắt các negatives).

Cách dùng
---------
    # Chạy với tham số mặc định (đọc data/raw, ghi data/processed)
    python data_engineering/clean_and_filter.py

    # Chỉ phân tích + chia tập, không copy ảnh (ghi dry_run_report.json +
    # dry_run_manifest.csv, KHÔNG đụng tới report/manifest của lần xuất thật)
    python data_engineering/clean_and_filter.py --dry-run

    # Tuỳ chỉnh thư mục, tỉ lệ chia, kích thước khối, tỉ lệ negatives, seed
    python data_engineering/clean_and_filter.py --raw data/raw --out data/processed \\
        --ratios 0.8 0.1 0.1 --chunk-size 25 --purge-gap 2 --neg-ratio 0.10 --seed 42

Chỉ dùng thư viện chuẩn + opencv-python + numpy + pyyaml.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

import cv2
import numpy as np
import yaml

# ============================================================================
# Hằng số & cấu hình mặc định
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_YAML_NAMES = ("data.yaml", "dataset.yaml")

# Các class được GIỮ LẠI, giữ nguyên ID gốc (0..9). Không bao giờ remap.
KEEP_CLASSES: tuple[int, ...] = tuple(range(10))
NUM_KEEP = len(KEEP_CLASSES)

SPLIT_NAMES: tuple[str, ...] = ("Train", "Val", "Test")

# Tên class dự phòng khi không tìm thấy data.yaml trong thư mục raw.
FALLBACK_CLASS_NAMES: dict[int, str] = {
    0: "Crosswalk sign", 1: "Highway entrance sign", 2: "Highway exit sign",
    3: "No-entry road sign", 4: "One-way road sign", 5: "Parking sign",
    6: "Priority sign", 7: "Roundabout sign", 8: "Stop sign", 9: "Pedestrian",
    10: "Car", 11: "Green Light", 12: "Red Light", 13: "Yellow Light",
    14: "Closed road stand", 15: "Parking spot", 16: "Stop line",
}

JPEG_SOI = b"\xff\xd8"  # Start Of Image
JPEG_EOI = b"\xff\xd9"  # End Of Image - JPEG hợp lệ phải kết thúc bằng 2 byte này
REPAIR_JPEG_QUALITY = 95
PROGRESS_EVERY = 2000
REVIEW_SHEET_SIZE = 48          # số negatives in lên ảnh ghép kiểm tra
REVIEW_TILE_WH = (240, 160)     # kích thước mỗi ô (W, H) trong ảnh ghép
LARGE_GROUP_FACTOR = 5          # cảnh báo nếu 1 nhóm có > 5 * chunk_size ảnh

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class CleanConfig:
    """Toàn bộ tham số của pipeline (được ghi lại nguyên văn vào report)."""

    raw_dir: Path = DEFAULT_RAW_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    chunk_size: int = 25
    purge_gap: int = 2
    neg_ratio: float = 0.10
    seed: int = 42
    clip_tol: float = 0.01
    min_box_px: float = 2.0
    max_aspect: float = 20.0
    workers: int = min(16, (os.cpu_count() or 4) + 4)
    dry_run: bool = False


# ============================================================================
# Cấu trúc dữ liệu
# ============================================================================
BoxTuple = tuple[int, float, float, float, float]  # (cls, cx, cy, w, h) chuẩn hoá


@dataclass
class LabelStats:
    """Bộ đếm cho bước làm sạch nhãn (C) và lọc lớp (D)."""

    lines_total: int = 0
    drop_reasons: Counter = field(default_factory=Counter)
    invalid_by_class: Counter = field(default_factory=Counter)
    kept_by_class: Counter = field(default_factory=Counter)
    class_filtered_by_class: Counter = field(default_factory=Counter)
    clipped: int = 0
    polygon_converted: int = 0

    def merge(self, other: "LabelStats") -> None:
        """Cộng dồn bộ đếm của một ảnh vào bộ đếm tổng."""
        self.lines_total += other.lines_total
        self.drop_reasons.update(other.drop_reasons)
        self.invalid_by_class.update(other.invalid_by_class)
        self.kept_by_class.update(other.kept_by_class)
        self.class_filtered_by_class.update(other.class_filtered_by_class)
        self.clipped += other.clipped
        self.polygon_converted += other.polygon_converted


@dataclass
class ImageRecord:
    """Thông tin của một ảnh xuyên suốt pipeline."""

    src: Path
    rel: str                      # đường dẫn tương đối (posix) so với raw_dir
    orig_split: str               # thư mục split gốc (train/valid/test) - chỉ để báo cáo
    label_path: Path | None
    source_key: str               # khoá frame gốc suy ra từ tên file
    size_bytes: int = 0
    md5: str = ""
    width: int = 0
    height: int = 0
    status: str = "pending"       # ok | zero_byte | corrupted
    truncated: bool = False       # JPEG thiếu EOI, chỉ decode được sau khi nối EOI -> sẽ sửa
    label_state: str = ""         # no_label_file | empty_label_file | has_lines
    boxes: list[BoxTuple] = field(default_factory=list)  # box class 0..9 hợp lệ
    invalid_target_lines: int = 0  # dòng class 0..9 (hoặc không rõ class) bị loại vì lỗi
    category: str = ""            # positive | negative_pool | ambiguous | suspect_negative
    group_key: str = ""           # khoá nhóm sau khi hợp nhất theo md5
    split: str = ""               # Train | Val | Test | "" (bị loại)
    chunk_id: int = -1
    out_name: str = ""


@dataclass
class Group:
    """Một nhóm = tất cả ảnh sinh ra từ CÙNG một frame gốc (kể cả bản augment)."""

    key: str
    family: str
    records: list[ImageRecord]

    @property
    def n_images(self) -> int:
        return len(self.records)

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(NUM_KEEP, dtype=np.int64)
        for rec in self.records:
            for box in rec.boxes:
                counts[box[0]] += 1
        return counts


# ============================================================================
# Tiện ích chung
# ============================================================================
_DIGITS_RE = re.compile(r"(\d+)")


def natural_key(text: str) -> tuple:
    """Khoá sắp xếp alpha-numeric: "frame_2" < "frame_10" (so số theo giá trị).

    Tách chuỗi thành các đoạn chữ / số xen kẽ, đoạn số được đổi sang int.
    re.split với nhóm bắt luôn trả về [chữ, số, chữ, số, ..., chữ] nên các phần
    tử cùng vị trí luôn cùng kiểu -> so sánh tuple an toàn.
    """
    parts = _DIGITS_RE.split(text.lower())
    return tuple(int(p) if i % 2 else p for i, p in enumerate(parts))


def run_parallel(func: Callable[[T], R], items: Sequence[T], workers: int,
                 tag: str) -> list[R]:
    """Chạy ``func`` song song bằng ThreadPoolExecutor, giữ nguyên thứ tự kết quả.

    cv2 và thao tác đọc/ghi file nhả GIL nên dùng thread là đủ nhanh, không cần
    multiprocessing (tránh chi phí pickle trên Windows).
    """
    results: list[R] = []
    total = len(items)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, res in enumerate(pool.map(func, items), start=1):
            results.append(res)
            if i % PROGRESS_EVERY == 0 or i == total:
                print(f"  [{tag}] {i}/{total}", flush=True)
    return results


def to_display_path(path: Path) -> str:
    """Đường dẫn posix tương đối so với PROJECT_ROOT (nếu được) để report gọn."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def class_name(names: dict[int, str], cls: int) -> str:
    return names.get(cls, f"class_{cls}")


# ============================================================================
# Bước A - Quét dữ liệu & ghép nhãn
# ============================================================================
_RF_SUFFIX_RE = re.compile(r"\.rf\.[0-9a-f]{8,}$", re.IGNORECASE)
_EXT_TOKEN_RE = re.compile(r"_(jpe?g|png|bmp|webp)$", re.IGNORECASE)
_COPY_WORD_RE = re.compile(
    r"[\s_-]+(?:kopya|copy|kopie|copie)(?:\s*\(\d+\))?$", re.IGNORECASE
)
_COPY_NUMBER_RE = re.compile(r"\s*\(\d+\)$")


def source_key_from_name(filename: str) -> str:
    """Suy ra khoá FRAME GỐC từ tên file Roboflow.

    Ví dụ:
        frame_0492_jpg.rf.b7170b....jpg          -> frame_0492_jpg
        yaya3_mp4-12-Kopya_jpg.rf.a37....jpg     -> yaya3_mp4-12_jpg
        image_20251219_095228_0006.jpg           -> image_20251219_095228_0006

    - ``.rf.<hex>``: hậu tố Roboflow thêm vào cho MỖI bản augment -> phải bỏ để
      mọi bản sao của cùng một frame có chung một khoá.
    - ``-Kopya`` / `` - Copy`` / ``(2)``: file bị nhân bản thủ công -> cùng frame.
    - ``_jpg`` / ``_png``: Roboflow mã hoá phần mở rộng gốc vào tên; ta giữ lại
      (frame_12_jpg và frame_12_png thường là 2 nguồn quay khác nhau).
    """
    stem = _RF_SUFFIX_RE.sub("", Path(filename).stem)
    match = _EXT_TOKEN_RE.search(stem)
    ext_token = match.group(1).lower() if match else ""
    if match:
        stem = stem[: match.start()]
    while True:  # có thể có nhiều hậu tố copy chồng nhau
        stripped = _strip_copy_suffix(stem)
        if stripped == stem:
            break
        stem = stripped
    stem = stem.lower()
    return f"{stem}_{ext_token}" if ext_token else stem


def _strip_copy_suffix(stem: str) -> str:
    """Bỏ MỘT hậu tố "bản sao" ở cuối stem (nếu có).

    - Có từ khoá copy ("-Kopya", " - Copy (2)"...) -> luôn bỏ.
    - Chỉ có số trong ngoặc "(n)" -> chỉ bỏ khi phần còn lại vẫn chứa chữ số
      ("frame_12 (2)" -> "frame_12"). Tên chung chung như "image (3)" /
      "image (4)" thường là các ảnh KHÁC nhau (điện thoại, WhatsApp đặt tên kiểu
      này) -> giữ nguyên để không gộp tất cả thành một nhóm khổng lồ "image".
      Bản sao trùng byte thật sự vẫn được gộp nhờ md5 ở bước F.
    """
    stripped = _COPY_WORD_RE.sub("", stem).rstrip(" _-")
    if stripped and stripped != stem:
        return stripped
    stripped = _COPY_NUMBER_RE.sub("", stem).rstrip(" _-")
    if stripped != stem and _DIGITS_RE.search(stripped):
        return stripped
    return stem


def family_of(key: str) -> str:
    """"Họ" tên file = khoá với mọi cụm số thay bằng '#'.

    frame_0492_jpg -> frame_#_jpg ; real_track_5_left_mp4-117_jpg -> real_track_#_left_mp#-#_jpg
    Dùng làm tiêu chí sắp xếp đầu tiên để các video/nguồn khác nhau KHÔNG bị
    xen kẽ nhau khi sắp xếp alpha-numeric.
    """
    return _DIGITS_RE.sub("#", key)


def discover_dataset(raw_dir: Path) -> tuple[list[ImageRecord], int]:
    """Quét đệ quy raw_dir, trả về danh sách ảnh (đã ghép nhãn) và số nhãn mồ côi.

    Quy ước ghép nhãn YOLO: thay thành phần "images" CUỐI CÙNG trong đường dẫn bằng
    "labels" và đổi đuôi thành .txt; nếu không có thì thử file .txt cùng thư mục.
    """
    image_paths: list[Path] = []
    txt_files: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(raw_dir):
        for name in filenames:
            path = Path(dirpath) / name
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                image_paths.append(path)
            elif suffix == ".txt":
                txt_files.add(os.path.normcase(str(path)))

    image_paths.sort(key=lambda p: natural_key(p.relative_to(raw_dir).as_posix()))
    used_labels: set[str] = set()
    records: list[ImageRecord] = []
    for img in image_paths:
        label = _find_label(img, txt_files)
        if label is not None:
            used_labels.add(os.path.normcase(str(label)))
        records.append(ImageRecord(
            src=img,
            rel=img.relative_to(raw_dir).as_posix(),
            orig_split=img.parent.name,
            label_path=label,
            source_key=source_key_from_name(img.name),
        ))

    # Nhãn mồ côi: file .txt nằm trong thư mục "labels" nhưng không có ảnh tương ứng.
    orphan = sum(
        1 for t in txt_files
        if t not in used_labels and "labels" in (p.lower() for p in Path(t).parts)
    )
    return records, orphan


def _find_label(img: Path, txt_files: set[str]) -> Path | None:
    parts = list(img.parts)
    lowered = [p.lower() for p in parts[:-1]]
    if "images" in lowered:
        idx = len(lowered) - 1 - lowered[::-1].index("images")
        candidate = Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")
        if os.path.normcase(str(candidate)) in txt_files:
            return candidate
    candidate = img.with_suffix(".txt")
    return candidate if os.path.normcase(str(candidate)) in txt_files else None


def load_class_names(raw_dir: Path) -> tuple[dict[int, str], str]:
    """Đọc "names" từ data.yaml/dataset.yaml đầu tiên tìm thấy trong raw_dir."""
    candidates = sorted(
        (p for name in DATASET_YAML_NAMES for p in raw_dir.rglob(name)),
        key=lambda p: natural_key(p.as_posix()),
    )
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
        names = data.get("names")
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}, to_display_path(path)
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}, to_display_path(path)
    return dict(FALLBACK_CLASS_NAMES), "fallback (hard-coded)"


# ============================================================================
# Bước B + C + D - Kiểm tra ảnh, làm sạch nhãn, lọc lớp (chạy song song mỗi ảnh)
# ============================================================================
def inspect_record(rec: ImageRecord, cfg: CleanConfig) -> LabelStats:
    """Kiểm tra 1 ảnh (bước B) rồi làm sạch + lọc nhãn của nó (bước C, D).

    Ghi kết quả trực tiếp vào ``rec`` và trả về bộ đếm nhãn của ảnh này.
    """
    stats = LabelStats()
    rec.size_bytes = rec.src.stat().st_size
    if rec.size_bytes == 0:
        rec.status = "zero_byte"
        return stats

    # Đọc bằng np.fromfile + cv2.imdecode: an toàn với đường dẫn Unicode trên Windows.
    buf = np.fromfile(str(rec.src), dtype=np.uint8)
    rec.md5 = hashlib.md5(buf).hexdigest()
    image, rec.truncated = decode_image(buf)
    if image is None:
        rec.status = "corrupted"
        return stats
    rec.height, rec.width = image.shape[:2]
    rec.status = "ok"
    _clean_labels(rec, cfg, stats)
    return stats


def decode_image(buf: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Decode ảnh từ bytes; trả về (ảnh hoặc None nếu hỏng, cờ JPEG bị cắt cụt).

    JPEG bị cắt cụt (ghi file dở dang, tải lỗi) thiếu marker EOI (FFD9) ở cuối.
    OpenCV 5 từ chối decode các file này. Khi lần decode đầu thất bại, ta thử
    thêm một lần với EOI được nối vào cuối: nếu thành công thì ảnh "cứu được"
    -> đánh dấu ``truncated`` để encode lại khi xuất (giống Ultralytics:
    "corrupt JPEG restored and saved"); nếu không -> ảnh hỏng.

    Cờ ``truncated`` CHỈ bật khi thật sự cần nối EOI mới decode được. Không suy
    ra từ "đuôi file thiếu FFD9" vì JPEG hợp lệ có thể có dữ liệu phụ (trailer
    của camera/điện thoại) sau EOI -> không được encode lại (mất chất lượng).
    """
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    truncated = False
    if image is None and buf[:2].tobytes() == JPEG_SOI \
            and not buf[-2:].tobytes() == JPEG_EOI:
        image = cv2.imdecode(np.concatenate([buf, np.frombuffer(JPEG_EOI, np.uint8)]),
                             cv2.IMREAD_COLOR)
        truncated = image is not None
    if image is None or image.size == 0:
        return None, False
    return image, truncated


def _read_label_lines(rec: ImageRecord) -> list[str]:
    if rec.label_path is None:
        rec.label_state = "no_label_file"
        return []
    text = rec.label_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rec.label_state = "has_lines" if lines else "empty_label_file"
    return lines


def _clean_labels(rec: ImageRecord, cfg: CleanConfig, stats: LabelStats) -> None:
    """Bước C (kiểm tra hình học) + D (lọc lớp 0..9, KHÔNG re-index)."""
    seen: set[str] = set()
    for line in _read_label_lines(rec):
        stats.lines_total += 1
        cls, box, reason = parse_label_line(line, rec.width, rec.height, cfg, stats)
        if reason is None:
            line_out = format_box(box)
            if line_out in seen:
                reason = "duplicate_line"
            else:
                seen.add(line_out)
        if reason is not None:
            stats.drop_reasons[reason] += 1
            stats.invalid_by_class["unknown" if cls is None else cls] += 1
            if reason != "duplicate_line" and (cls is None or cls in KEEP_CLASSES):
                # Một đối tượng class 0..9 (hoặc không rõ class) có thật trong ảnh
                # nhưng mất nhãn -> dùng để phát hiện ảnh "mơ hồ" / "thiếu nhãn"
                # ở bước E. Dòng trùng lặp không tính (đối tượng vẫn còn nhãn).
                rec.invalid_target_lines += 1
            continue
        # ---- Bước D: LỌC LỚP. Giữ nguyên class_id 0..9, bỏ mọi class khác. ----
        if cls in KEEP_CLASSES:
            rec.boxes.append(box)
            stats.kept_by_class[cls] += 1
        else:
            stats.class_filtered_by_class[cls] += 1


def parse_label_line(line: str, width: int, height: int, cfg: CleanConfig,
                     stats: LabelStats) -> tuple[int | None, BoxTuple | None, str | None]:
    """Phân tích + kiểm tra 1 dòng nhãn YOLO.

    Trả về (class_id hoặc None, box chuẩn hoá đã clip hoặc None, lý do loại hoặc None).
    Hỗ trợ cả dòng polygon "cls x1 y1 x2 y2 ..." (số trường lẻ > 5) -> đổi sang bbox.
    """
    parts = line.split()
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return _int_or_none(parts[0]), None, "non_numeric"
    if not all(math.isfinite(v) for v in values):
        return _int_or_none(parts[0]), None, "non_finite"
    if values[0] < 0 or values[0] != int(values[0]):
        return None, None, "bad_class_id"
    cls = int(values[0])

    if len(values) == 5:
        cx, cy, w, h = values[1:]
    elif len(values) >= 7 and len(values) % 2 == 1:
        xs, ys = values[1::2], values[2::2]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        stats.polygon_converted += 1
    else:
        return cls, None, "wrong_field_count"

    # Đổi sang toạ độ pixel thật của ảnh để kiểm tra chính xác.
    bw, bh = w * width, h * height
    if bw <= 0 or bh <= 0:
        return cls, None, "non_positive_area"
    pcx, pcy = cx * width, cy * height
    if not (0 <= pcx <= width and 0 <= pcy <= height):
        return cls, None, "center_outside_image"
    x1, y1, x2, y2 = pcx - bw / 2, pcy - bh / 2, pcx + bw / 2, pcy + bh / 2
    tol_x, tol_y = cfg.clip_tol * width, cfg.clip_tol * height
    if x1 < -tol_x or y1 < -tol_y or x2 > width + tol_x or y2 > height + tol_y:
        return cls, None, "beyond_image_border"

    # Lệch biên trong dung sai = sai số làm tròn của công cụ gán nhãn -> clip.
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(width), x2), min(float(height), y2)
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        stats.clipped += 1
    bw, bh = cx2 - cx1, cy2 - cy1
    if bw < cfg.min_box_px or bh < cfg.min_box_px:
        return cls, None, "too_small"
    if max(bw / bh, bh / bw) > cfg.max_aspect:
        return cls, None, "extreme_aspect_ratio"
    box = (cls, (cx1 + cx2) / 2 / width, (cy1 + cy2) / 2 / height,
           bw / width, bh / height)
    return cls, box, None


def _int_or_none(token: str) -> int | None:
    try:
        value = float(token)
    except ValueError:
        return None
    return int(value) if math.isfinite(value) and value == int(value) else None


def format_box(box: BoxTuple) -> str:
    return "%d %.6f %.6f %.6f %.6f" % box


# ============================================================================
# Bước E - Background Negatives
# ============================================================================
# ---------------------------------------------------------------------------
# GIẢI THÍCH CHI TIẾT: VÌ SAO CẦN (VÀ CHỈ CẦN MỘT ÍT) BACKGROUND NEGATIVES?
# ---------------------------------------------------------------------------
# "Background negative" là ảnh KHÔNG chứa đối tượng nào thuộc 10 class cần nhận
# diện; file nhãn tương ứng là file .txt RỖNG. Với YOLO, ảnh rỗng vẫn đóng góp vào
# loss: mọi anchor/điểm dự đoán trên ảnh đều bị phạt nếu báo "có vật thể"
# (objectness/classification loss với nhãn 0).
#
# 1) Vì sao CẦN negatives -> giảm False Positive (báo nhầm):
#    - Trên sa bàn có rất nhiều thứ "trông giống" biển báo: đèn giao thông, xe,
#      cột chắn đường, vạch dừng, ô đỗ xe, đồ vật màu đỏ/xanh... Nếu model chưa
#      từng thấy những cảnh này mà không có nhãn, nó dễ "đoán bừa" là Stop sign
#      hay Crosswalk sign.
#    - Khi xe RC chạy thật, phần lớn thời gian camera nhìn đoạn đường TRỐNG.
#      Một False Positive (ví dụ báo "Stop sign" giả) có thể khiến xe phanh gấp.
#    - Ảnh chỉ chứa Car / đèn giao thông / Stop line... (class 10..16 đã bị lọc
#      bỏ) là "hard negatives" rất tốt: chúng dạy model rằng "đèn đỏ KHÔNG phải
#      biển Stop", "xe KHÔNG phải người đi bộ".
#
# 2) Vì sao KHÔNG giữ hết (ở đây pool negatives ~ một nửa bộ dữ liệu!):
#    - Mất cân bằng object-vs-background: YOLO vốn đã có hàng nghìn ô lưới là
#      nền trên MỖI ảnh, chỉ vài ô chứa vật thể. Thêm quá nhiều ảnh rỗng làm tín
#      hiệu "không có gì" lấn át -> model trở nên "nhút nhát", hạ confidence ->
#      giảm Recall (bỏ sót biển báo thật) - lỗi nguy hiểm hơn với xe tự hành.
#    - Lãng phí tài nguyên: mỗi epoch tốn thời gian cho ảnh gần như không cung
#      cấp gradient hữu ích về hình dạng biển báo/người đi bộ.
#    - Roboflow đã nhân bản mỗi ảnh nền 1..21 lần (augment) -> rất nhiều ảnh nền
#      gần như trùng nhau, giữ hết chỉ làm "phình" dữ liệu.
#
# 3) Vì sao 10%:
#    - Khuyến nghị thực nghiệm của Ultralytics: ảnh nền nên chiếm khoảng 0-10%
#      bộ dữ liệu để giảm FP mà không hại Recall (COCO có ~1% ảnh nền).
#    - Giữ 10% pool cho ra tỉ lệ negatives cuối cùng xấp xỉ 10% tổng số ảnh
#      (được in ra và ghi vào report để kiểm chứng). Tham số --neg-ratio cho
#      phép thử nghiệm (ablation) 0.05 / 0.10 / 0.20.
#
# 4) Ảnh "MƠ HỒ" (ambiguous) bị LOẠI khỏi pool:
#    Ảnh gốc CÓ box class 0..9 nhưng mọi box đó đều bị loại ở bước C (box lỗi).
#    Ảnh vẫn chứa biển báo/người thật nhưng không còn nhãn -> nếu dùng làm
#    negative sẽ dạy model "bỏ qua biển báo thật" => tuyệt đối không dùng.
#    (Ảnh còn ÍT NHẤT một box hợp lệ nhưng mất một box class 0..9 khác thì vẫn
#    là positive; chúng được đếm/liệt kê là "partially_labeled" trong report để
#    kiểm tra thủ công.)
#
# 5) Ảnh "ĐÁNG NGỜ" (suspect) cũng bị LOẠI khỏi pool:
#    Roboflow export ảnh "null" (chưa gán nhãn hoặc bị đánh dấu là nền) KHÔNG có
#    file .txt. Kiểm tra bằng mắt cho thấy một phần các ảnh này vẫn chứa biển báo
#    (vd. biển người đi bộ) chỉ là chưa ai gán nhãn - nhất là khi bản sao khác
#    của CÙNG frame gốc (cùng source_key) lại có box class 0..9. Quy tắc:
#      - no_label_file + cùng frame gốc với một ảnh positive  -> suspect (loại);
#      - empty_label_file / only_non_target_classes            -> tin cậy (giữ):
#        người gán nhãn đã mở ảnh này và chủ động gán (hoặc không gán) nhãn.
#    Quy tắc này không bắt được 100% (ảnh null có biển báo nhưng không có bản
#    sao nào được gán nhãn), nên script xuất thêm ``negatives_review.jpg`` - ảnh
#    ghép ngẫu nhiên các negatives được giữ - để kiểm tra lại bằng mắt.
#
# 6) Cách chọn ngẫu nhiên (có seed, tái lập được) và ưu tiên ĐA DẠNG:
#    Các bản augment của cùng một frame gần như giống nhau, nên ta xáo trộn có
#    seed rồi lấy "bản thứ 1 của mọi frame" trước, sau đó mới tới "bản thứ 2"...
#    => cùng một số lượng ảnh nhưng phủ được nhiều cảnh nền khác nhau nhất.
#
# 7) Vì sao lấy mẫu SAU khi chia tập (và trong TỪNG split):
#    Toàn bộ pool tin cậy tham gia chia tập ở bước F cùng các ảnh positive; sau
#    đó mới giữ --neg-ratio số negatives của MỖI split (RNG riêng). Nhờ vậy:
#      - Việc ảnh positive nào vào Train/Val/Test KHÔNG phụ thuộc --neg-ratio
#        => các thí nghiệm ablation 0.05 / 0.10 / 0.20 dùng CHUNG một tập Test,
#        so sánh mAP mới có ý nghĩa.
#      - Tập negatives của tỉ lệ nhỏ là TẬP CON của tỉ lệ lớn (cùng thứ tự ưu
#        tiên) -> ablation "lồng nhau", chỉ khác đúng số ảnh nền.
#      - Mỗi split đều có ~--neg-ratio pool negatives của nó -> tỉ lệ ảnh nền
#        trong Train/Val/Test xấp xỉ nhau.
# ---------------------------------------------------------------------------
def categorize_record(rec: ImageRecord) -> str:
    """Phân loại ảnh hợp lệ: positive | negative_pool | ambiguous."""
    if rec.boxes:
        return "positive"
    if rec.invalid_target_lines > 0:
        return "ambiguous"
    return "negative_pool"


def is_partially_labeled(rec: ImageRecord) -> bool:
    """Positive nhưng có ít nhất một box class 0..9 bị loại vì lỗi (thiếu nhãn)."""
    return bool(rec.boxes) and rec.invalid_target_lines > 0


def flag_suspect_negatives(pool: Sequence[ImageRecord],
                           positives: Sequence[ImageRecord]) -> list[ImageRecord]:
    """Đánh dấu ``suspect_negative`` cho ảnh không có file nhãn nhưng cùng frame
    gốc với một ảnh positive; trả về pool tin cậy (đã bỏ các ảnh đáng ngờ)."""
    positive_sources = {rec.source_key for rec in positives}
    trusted: list[ImageRecord] = []
    for rec in pool:
        if rec.label_state == "no_label_file" and rec.source_key in positive_sources:
            rec.category = "suspect_negative"
        else:
            trusted.append(rec)
    return trusted


def negative_reason(rec: ImageRecord) -> str:
    """Lý do một ảnh rơi vào pool negatives (để thống kê)."""
    if rec.label_state == "has_lines":
        return "only_non_target_classes"
    return rec.label_state  # no_label_file | empty_label_file


def select_background_negatives(pool: Sequence[ImageRecord], ratio: float,
                                rng: random.Random) -> list[ImageRecord]:
    """Chọn ngẫu nhiên (có seed) round(ratio * |pool|) ảnh nền, ưu tiên đa dạng.

    Ảnh của cùng một frame gốc (source_key) được xáo trộn và đánh hạng 0, 1, 2...
    Sau đó sắp xếp theo (hạng, số ngẫu nhiên) và lấy phần đầu: mọi frame khác
    nhau được lấy 1 bản trước khi bất kỳ frame nào được lấy bản thứ 2.
    Thứ tự ưu tiên không phụ thuộc ``ratio`` -> kết quả của tỉ lệ nhỏ là tập con
    của tỉ lệ lớn. Pool khác rỗng và ratio > 0 thì giữ ít nhất 1 ảnh.
    """
    n_keep = 0
    if pool and ratio > 0:
        n_keep = min(len(pool), max(1, int(round(ratio * len(pool)))))
    by_source: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in pool:
        by_source[rec.source_key].append(rec)

    ranked: list[tuple[int, float, str, ImageRecord]] = []
    for key in sorted(by_source, key=natural_key):
        members = sorted(by_source[key], key=lambda r: natural_key(r.rel))
        rng.shuffle(members)
        for rank, rec in enumerate(members):
            ranked.append((rank, rng.random(), rec.rel, rec))
    ranked.sort(key=lambda item: item[:3])
    return [item[3] for item in ranked[:n_keep]]


def sample_negatives_per_split(pool: Sequence[ImageRecord], ratio: float,
                               rng: random.Random) -> list[ImageRecord]:
    """Sau bước F: trong MỖI split giữ ``ratio`` số ảnh nền của split đó.

    Ảnh nền không được chọn bị bỏ (``split = ""``), ảnh đã bị purge thì vốn
    không thuộc split nào. Trả về danh sách negatives được giữ.
    """
    kept: list[ImageRecord] = []
    for split in SPLIT_NAMES:
        members = [rec for rec in pool if rec.split == split]
        chosen = select_background_negatives(members, ratio, rng)
        chosen_ids = {id(rec) for rec in chosen}
        for rec in members:
            if id(rec) not in chosen_ids:
                rec.split = ""
        kept.extend(chosen)
    return kept


# ============================================================================
# Bước F - Sequence-based Split (chống rò rỉ dữ liệu)
# ============================================================================
# ---------------------------------------------------------------------------
# GIẢI THÍCH CHI TIẾT: VÌ SAO KHÔNG ĐƯỢC random.shuffle() TOÀN BỘ ẢNH?
# ---------------------------------------------------------------------------
# Dữ liệu được cắt từ VIDEO quay sa bàn. Hai frame liên tiếp chỉ cách nhau vài
# chục mili-giây -> gần như GIỐNG HỆT nhau (xe chỉ dịch vài cm):
#
#     Video:  f001 f002 f003 f004 f005 f006 f007 f008 f009 f010 ...
#             \____ cùng một cảnh, cùng một biển báo, cùng ánh sáng ____/
#
#   Random shuffle toàn bộ ảnh rồi cắt 80/10/10:
#     Train:  f001      f003 f004      f006 f007      f009 ...
#     Val  :       f002                          f008      ...
#     Test :                      f005                f010 ...
#              => f002 (Val) ~ f001 + f003 (Train): model đã "thấy" đáp án!
#
#   Hệ quả: mAP trên Val/Test cao ẢO (model học thuộc cảnh chứ không học đặc
#   trưng biển báo), chọn sai hyper-parameter / early-stopping, và khi xe RC chạy
#   thực tế (cảnh mới, góc mới, ánh sáng mới) độ chính xác tụt mạnh.
#
#   Tệ hơn nữa, Roboflow đã tạo 1..21 bản augment của CÙNG một frame gốc:
#     frame_0492_jpg.rf.b717....jpg  -> Train
#     frame_0492_jpg.rf.4c2e....jpg  -> Test    <- cùng một ảnh, chỉ khác augment!
#   (Đã kiểm tra: ~850 frame gốc xuất hiện ở nhiều split của Roboflow => split gốc
#   bị rò rỉ, vì vậy ta CHIA LẠI từ đầu.)
#
# CÁCH LÀM Ở ĐÂY (sequence-based / grouped split):
#   1. GOM NHÓM: mọi ảnh của cùng một frame gốc (bỏ ".rf.<hash>", "-Kopya",
#      " - Copy"...) thành MỘT nhóm. Ảnh trùng byte (cùng md5) nhưng khác tên
#      cũng bị gộp chung nhóm (union-find) -> chắc chắn không rơi vào 2 split.
#   2. SẮP XẾP alpha-numeric các nhóm (theo "họ" tên file, rồi natural sort):
#      frame_2 < frame_10 < frame_100 => các frame liên tiếp của một video nằm
#      CẠNH NHAU trong danh sách. Không dùng random.shuffle trên ảnh.
#   3. CẮT KHỐI (chunk): danh sách nhóm đã sắp xếp được cắt thành các khối gồm
#      --chunk-size nhóm LIÊN TIẾP (mặc định 25):
#
#        [ f001..f025 ][ f026..f050 ][ f051..f075 ][ f076..f100 ][ ... ]
#           chunk 0       chunk 1       chunk 2       chunk 3
#            Train         Train          Val          Train
#                              ^purge^         ^purge^
#
#      Vì sao 25? Positives + pool negatives tin cậy tạo ~10.500 nhóm -> ~420
#      khối (~55 ảnh/khối, cỡ vài giây video). Đã thử nghiệm: sai lệch tỉ lệ
#      class lớn nhất SAU purge (seed 42 | trung bình 8 seed) và số ảnh positive
#      bị purge (seed 42):
#          chunk=5 : 0.49 | -    điểm %, purge 1501 ảnh (quá nhiều biên -> mất dữ liệu)
#          chunk=10: 0.08 | 0.16 điểm %, purge  786 ảnh
#          chunk=15: 0.12 | 0.17 điểm %, purge  505 ảnh
#          chunk=20: 0.24 | 0.23 điểm %, purge  371 ảnh
#          chunk=25: 0.17 | 0.22 điểm %, purge  321 ảnh  <- điểm cân bằng
#          chunk=30: 0.80 | 0.72 điểm %, purge  304 ảnh
#          chunk=40: 0.43 | 0.53 điểm %, purge  228 ảnh
#          chunk=60: 2.46 | -    điểm %, purge  112 ảnh (quá ít khối -> khó phân tầng)
#      Khối nhỏ => nhiều biên giữa các split => phải purge nhiều + tăng nguy cơ
#      2 frame gần nhau nằm ở 2 split; khối lớn => Val/Test chỉ gồm vài khối,
#      ít đa dạng và khó đạt tỉ lệ 80/10/10 cho từng class. 10-15 chỉ tốt hơn 25
#      chưa tới 0.1 điểm % nhưng mất gấp 1.6-2.4 lần dữ liệu ở biên.
#   4. GÁN NGUYÊN KHỐI vào Train/Val/Test theo tỉ lệ 80/10/10 với thuật toán
#      tham lam PHÂN TẦNG (iterative stratification, Sechidis et al. 2011, phiên
#      bản cho "khối"), sau đó tinh chỉnh bằng local search (move/swap khối):
#      mỗi lớp 0..9, số ảnh positive và số ảnh nền đều được chia xấp xỉ
#      80/10/10, tránh trường hợp Test không có (hoặc có quá ít) một loại biển báo.
#   5. PURGE GAP: tại mỗi biên mà 2 khối kề nhau thuộc 2 split khác nhau, bỏ
#      --purge-gap nhóm ở phía split lớn hơn (Train) -> frame cuối khối Train và
#      frame đầu khối Val/Test không còn là 2 frame sát nhau của cùng video.
#      Vì purge "cắt" bớt Train ở biên, phép gán được tinh chỉnh thêm vài vòng
#      với đặc trưng khối ĐÃ TRỪ nhóm bị purge (purge-aware refine) để tỉ lệ
#      thực tế sau purge vẫn sát 80/10/10.
#   6. KIỂM CHỨNG: không có group_key, source_key hay md5 nào ở hơn 1 split.
#
# GIỚI HẠN (nên nêu trong báo cáo):
#   - Nhóm dựa trên TÊN file (Roboflow) + md5 trùng byte. Hai ảnh gần giống nhau
#     nhưng khác tên và khác byte KHÔNG được phát hiện: ví dụ họ
#     image_YYYYMMDD_HHMMSS_NNNN (~1 frame/giây, xe đứng yên -> các frame cách
#     nhau 3-4 số gần như y hệt) hoặc đường đua vòng lặp đi qua cùng cảnh ở các
#     số frame khác nhau. Purge gap chỉ che 2 nhóm kề biên. Đo bằng dHash 256-bit:
#     ~0.1% ảnh Val/Test có ảnh Train rất gần (Hamming <= 4).
#   - Ngược lại, một số tên frame (vd. "frame207_jpg") được dùng cho nhiều cảnh
#     khác nhau từ các video khác nhau -> bị gộp chung nhóm; điều này chỉ làm
#     phép chia "thận trọng" hơn, không gây rò rỉ.
# ---------------------------------------------------------------------------
def build_groups(records: Sequence[ImageRecord]) -> list[Group]:
    """Gom ảnh theo frame gốc + hợp nhất các nhóm có ảnh trùng md5 (union-find).

    Trả về danh sách nhóm đã sắp xếp theo (họ tên file, natural sort).
    """
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:  # nén đường đi
            parent[key], key = root, parent[key]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Gốc là khoá đứng trước theo thứ tự sắp xếp -> kết quả tất định.
            first, second = sorted((ra, rb), key=_group_sort_key)
            parent[second] = first

    for rec in records:
        parent.setdefault(rec.source_key, rec.source_key)
    first_key_by_md5: dict[str, str] = {}
    for rec in records:
        other = first_key_by_md5.setdefault(rec.md5, rec.source_key)
        union(other, rec.source_key)

    members: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        rec.group_key = find(rec.source_key)
        members[rec.group_key].append(rec)

    groups = [
        Group(key=key, family=family_of(key),
              records=sorted(recs, key=lambda r: natural_key(r.rel)))
        for key, recs in members.items()
    ]
    groups.sort(key=lambda g: _group_sort_key(g.key))
    return groups


def _group_sort_key(key: str) -> tuple:
    """Sắp xếp theo họ tên file trước, rồi alpha-numeric trong họ."""
    return natural_key(family_of(key)), natural_key(key)


def make_chunks(n_groups: int, chunk_size: int) -> list[list[int]]:
    """Cắt danh sách nhóm (đã sắp xếp) thành các khối chỉ số liên tiếp."""
    return [list(range(i, min(i + chunk_size, n_groups)))
            for i in range(0, n_groups, chunk_size)]


def chunk_features(groups: Sequence[Group], chunks: Sequence[list[int]],
                   skip: frozenset[int] = frozenset()) -> np.ndarray:
    """Ma trận đặc trưng mỗi khối (n_chunks x 12), bỏ qua các nhóm trong ``skip``.

    Cột 0..9: số box của từng class; cột 10: số ảnh nền (pool negatives); cột 11
    (cuối): số ảnh positive. Nhờ 2 cột cuối, số ảnh positive lẫn số ảnh nền (và
    do đó cả tổng số ảnh sau khi lấy mẫu negatives) cũng được chia ~80/10/10.
    ``skip`` = các nhóm bị purge -> đặc trưng "thực tế" của khối sau purge.
    """
    feats = np.zeros((len(chunks), NUM_KEEP + 2), dtype=np.float64)
    for c, idxs in enumerate(chunks):
        for gi in idxs:
            if gi in skip:
                continue
            group = groups[gi]
            n_neg = sum(1 for r in group.records if not r.boxes)
            feats[c, :NUM_KEEP] += group.class_counts()
            feats[c, NUM_KEEP] += n_neg
            feats[c, -1] += group.n_images - n_neg
    return feats


def assign_chunks_stratified(feats: np.ndarray, ratios: Sequence[float],
                             rng: random.Random) -> list[int]:
    """Gán từng khối vào split bằng iterative stratification (tham lam).

    Ý tưởng (Sechidis et al., 2011 - áp dụng cho khối thay vì từng ảnh):
      - "Nhu cầu" của split s với chiều j (class j hoặc số ảnh):
            demand[s, j] = ratio[s] * tổng[j]
      - Duyệt các khối theo thứ tự: nhiều box trước (khó xếp nhất), hoà thì theo
        số ngẫu nhiên có seed. Khối lớn được đặt trước, khối nhỏ "lấp chỗ trống"
        ở cuối giúp tỉ lệ cuối cùng rất sát mục tiêu.
      - Với mỗi khối, điểm của split s = trung bình có trọng số của nhu cầu CÒN
        LẠI tương đối (remaining / demand) trên các chiều mà khối có. Trọng số
        của chiều j = (số lượng của khối ở chiều j) / tổng[j] -> class hiếm có
        trọng số lớn hơn (giống việc Sechidis ưu tiên nhãn hiếm).
      - Khối được gán cho split có điểm CAO NHẤT (split "đói" nhất với đúng các
        class mà khối chứa). Hoà điểm -> split còn thiếu nhiều ảnh positive hơn.
    """
    ratios_arr = np.asarray(ratios, dtype=np.float64)
    totals = feats.sum(axis=0)
    safe_totals = np.where(totals > 0, totals, 1.0)
    demand = ratios_arr[:, None] * totals[None, :]
    safe_demand = np.where(demand > 0, demand, 1.0)
    assigned = np.zeros_like(demand)

    tie_breakers = [rng.random() for _ in range(len(feats))]
    order = sorted(range(len(feats)),
                   key=lambda c: (-feats[c, :NUM_KEEP].sum(), tie_breakers[c]))
    result = [-1] * len(feats)
    for c in order:
        weights = feats[c] / safe_totals
        best_split, best_key = -1, None
        for s in range(len(ratios_arr)):
            if ratios_arr[s] <= 0:
                continue
            remaining = (demand[s] - assigned[s]) / safe_demand[s]
            score = float((weights * remaining).sum() / max(weights.sum(), 1e-12))
            key = (score, demand[s, -1] - assigned[s, -1])
            if best_key is None or key > best_key:
                best_split, best_key = s, key
        assigned[best_split] += feats[c]
        result[c] = best_split
    return result


def refine_assignment(feats: np.ndarray, assignment: Sequence[int],
                      ratios: Sequence[float], max_iter: int = 2000,
                      max_swap_chunks: int = 2000) -> tuple[list[int], int]:
    """Tinh chỉnh kết quả tham lam bằng tìm kiếm cục bộ (local search).

    Hàm mục tiêu = tổng bình phương sai lệch (assigned - demand) / tổng, trên mọi
    split và mọi chiều (đúng thước đo "điểm %" được báo cáo). Mỗi vòng thử TẤT CẢ
    các phép:
      - CHUYỂN (move) một khối sang split khác,
      - ĐỔI CHỖ (swap) hai khối thuộc hai split khác nhau,
    rồi áp dụng phép giảm mục tiêu nhiều nhất; dừng khi không còn phép nào cải
    thiện. Vẫn chỉ di chuyển NGUYÊN KHỐI nên tính liên tục của chuỗi được giữ.
    Thuần numpy, tất định (argmin lấy chỉ số nhỏ nhất khi hoà). Khi số khối quá
    lớn (> max_swap_chunks, ví dụ --chunk-size 1) chỉ thử "move" để tiết kiệm RAM.

    Với r_s = (A_s - D_s) / T và d = vector đặc trưng khối / T:
      move c: s1 -> s2  : Δ = Σ 2d(r_s2 - r_s1) + 2d²
      swap c1(s1)<->c2(s2): Δ = Σ 2δ(r_s1 - r_s2) + 2δ²,  δ = d_c2 - d_c1
    """
    if len(feats) == 0:
        return list(assignment), 0
    ratios_arr = np.asarray(ratios, dtype=np.float64)
    allowed = ratios_arr > 0
    totals = feats.sum(axis=0)
    scaled = feats / np.where(totals > 0, totals, 1.0)          # d của từng khối
    target = ratios_arr[:, None] * (totals > 0)[None, :]        # D / T
    assign = np.asarray(assignment, dtype=np.int64).copy()
    n_splits = len(ratios_arr)

    n_moves = 0
    for _ in range(max_iter):
        resid = np.stack([scaled[assign == s].sum(axis=0) for s in range(n_splits)]) - target
        own = resid[assign]                                     # r của split hiện tại (n x dims)
        sq = (scaled ** 2).sum(axis=1)

        # Move: delta[c, s] cho mọi khối c và split đích s.
        move = 2 * scaled @ resid.T - 2 * (scaled * own).sum(axis=1)[:, None] + 2 * sq[:, None]
        move[np.arange(len(assign)), assign] = np.inf
        move[:, ~allowed] = np.inf

        best_move, best_swap = float(move.min()), np.inf
        if len(assign) <= max_swap_chunks:
            # Swap: delta[c1, c2] khai triển thành các tích ma trận n x n.
            so = scaled @ own.T                                  # [c1, c2] = d_c1 · r_c2
            self_dot = (scaled * own).sum(axis=1)                # d_c · r_c
            cross = 2 * (so.T - self_dot[None, :] - self_dot[:, None] + so)
            swap = cross + 2 * (sq[:, None] + sq[None, :] - 2 * scaled @ scaled.T)
            swap[assign[:, None] == assign[None, :]] = np.inf
            best_swap = float(swap.min())
        if min(best_move, best_swap) >= -1e-12:
            break
        if best_move <= best_swap:
            c, s = np.unravel_index(int(np.argmin(move)), move.shape)
            assign[c] = s
        else:
            c1, c2 = np.unravel_index(int(np.argmin(swap)), swap.shape)
            assign[c1], assign[c2] = assign[c2], assign[c1]
        n_moves += 1
    return assign.tolist(), n_moves


def _max_deviation_pct(feats: np.ndarray, assignment: Sequence[int],
                       ratios: Sequence[float]) -> float:
    """Sai lệch tuyệt đối lớn nhất (điểm %) giữa tỉ lệ thực và mục tiêu (trước purge).

    Bỏ qua các chiều có tổng = 0 (class vắng mặt, không có negatives): không có
    gì để chia nên không thể coi là "lệch 80 điểm %".
    """
    totals = feats.sum(axis=0)
    present = totals > 0
    if not present.any():
        return 0.0
    assign = np.asarray(assignment)
    shares = np.stack([feats[assign == s].sum(axis=0) for s in range(len(ratios))])
    shares = shares[:, present] / totals[present]
    return round(float(np.abs(shares - np.asarray(ratios)[:, None]).max() * 100), 2)


def apply_purge_gap(chunks: Sequence[list[int]], chunk_split: Sequence[int],
                    gap: int, ratios: Sequence[float]) -> tuple[set[int], int]:
    """Bỏ ``gap`` nhóm ở biên giữa 2 khối kề nhau thuộc 2 split khác nhau.

    Nhóm bị bỏ nằm ở phía split có tỉ lệ lớn hơn (Train mất ít "giá trị" nhất);
    nếu bằng nhau (Val-Test) thì bỏ ở khối phía sau. Mỗi phía của một khối mất
    tối đa một nửa số nhóm của khối để không khối nào bị xoá hết.
    Trả về (tập chỉ số nhóm bị bỏ, số biên đã xử lý).
    """
    purged: set[int] = set()
    n_boundaries = 0
    if gap <= 0:
        return purged, n_boundaries
    for i in range(len(chunks) - 1):
        a, b = chunk_split[i], chunk_split[i + 1]
        if a == b:
            continue
        n_boundaries += 1
        side = i if ratios[a] > ratios[b] else i + 1
        groups = chunks[side]
        k = min(gap, len(groups) // 2)
        purged.update(groups[len(groups) - k:] if side == i else groups[:k])
    return purged, n_boundaries


def _post_purge_deviation(groups: Sequence[Group], chunks: Sequence[list[int]],
                          chunk_split: Sequence[int], cfg: CleanConfig) -> float:
    """Sai lệch max (điểm %) tính trên dữ liệu THỰC SỰ còn lại sau purge."""
    purged, _ = apply_purge_gap(chunks, chunk_split, cfg.purge_gap, cfg.ratios)
    return _max_deviation_pct(chunk_features(groups, chunks, frozenset(purged)),
                              chunk_split, cfg.ratios)


def purge_aware_refine(groups: Sequence[Group], chunks: Sequence[list[int]],
                       chunk_split: Sequence[int], cfg: CleanConfig,
                       max_rounds: int = 5) -> tuple[list[int], int]:
    """Tinh chỉnh lại phép gán khi đã biết các nhóm sẽ bị purge.

    Purge luôn "cắt" vài nhóm ở biên phía Train -> tỉ lệ thực tế lệch khỏi kết
    quả đã tối ưu. Mỗi vòng: tính purge của phép gán hiện tại, chạy lại local
    search trên đặc trưng khối ĐÃ TRỪ các nhóm bị purge, rồi đo lại sai lệch
    sau purge của phép gán mới. Giữ phép gán tốt nhất; dừng khi không cải thiện.
    Trả về (phép gán tốt nhất, số vòng đã cải thiện).
    """
    best = list(chunk_split)
    best_dev = _post_purge_deviation(groups, chunks, best, cfg)
    rounds = 0
    for _ in range(max_rounds):
        purged, _ = apply_purge_gap(chunks, best, cfg.purge_gap, cfg.ratios)
        effective = chunk_features(groups, chunks, frozenset(purged))
        candidate, _ = refine_assignment(effective, best, cfg.ratios)
        dev = _post_purge_deviation(groups, chunks, candidate, cfg)
        if dev >= best_dev:
            break
        best, best_dev, rounds = candidate, dev, rounds + 1
    return best, rounds


def sequence_split(records: Sequence[ImageRecord], cfg: CleanConfig,
                   rng: random.Random) -> dict:
    """Thực hiện toàn bộ bước F; gán ``rec.split`` / ``rec.chunk_id`` tại chỗ.

    ``records`` gồm ảnh positive + TOÀN BỘ pool negatives tin cậy (lấy mẫu sau).
    """
    groups = build_groups(records)
    chunks = make_chunks(len(groups), cfg.chunk_size)
    feats = chunk_features(groups, chunks)
    greedy_split = assign_chunks_stratified(feats, cfg.ratios, rng)
    refined_split, n_refine_steps = refine_assignment(feats, greedy_split, cfg.ratios)
    chunk_split, purge_rounds = purge_aware_refine(groups, chunks, refined_split, cfg)
    purged, n_boundaries = apply_purge_gap(chunks, chunk_split, cfg.purge_gap, cfg.ratios)

    purged_images = Counter()
    for c, idxs in enumerate(chunks):
        for gi in idxs:
            for rec in groups[gi].records:
                rec.chunk_id = c
                if gi in purged:
                    purged_images["positive" if rec.boxes else "negative_pool"] += 1
                else:
                    rec.split = SPLIT_NAMES[chunk_split[c]]

    largest = max(groups, key=lambda g: g.n_images)
    chunks_per_split = Counter(SPLIT_NAMES[s] for s in chunk_split)
    return {
        "n_groups": len(groups),
        "n_md5_merged_groups": sum(
            1 for g in groups if len({r.source_key for r in g.records}) > 1),
        "largest_group": {"key": largest.key, "images": largest.n_images},
        "n_chunks": len(chunks),
        "chunks_per_split": {s: chunks_per_split.get(s, 0) for s in SPLIT_NAMES},
        "max_abs_dev_pct_greedy": _max_deviation_pct(feats, greedy_split, cfg.ratios),
        "max_abs_dev_pct_refined": _max_deviation_pct(feats, refined_split, cfg.ratios),
        "refinement_steps": n_refine_steps,
        "max_abs_dev_pct_after_purge": _post_purge_deviation(groups, chunks, chunk_split, cfg),
        "purge_aware_rounds": purge_rounds,
        "n_split_boundaries": n_boundaries,
        "purged_groups": len(purged),
        "purged_positive_images": purged_images["positive"],
        "purged_negative_pool_images": purged_images["negative_pool"],
    }


# ============================================================================
# Kiểm chứng & thống kê split
# ============================================================================
def verify_no_leakage(records: Iterable[ImageRecord]) -> dict:
    """Kiểm tra không có group_key / md5 / source_key nào nằm ở > 1 split."""
    owners: dict[str, dict[str, set[str]]] = {
        "group_key": defaultdict(set), "md5": defaultdict(set), "source_key": defaultdict(set)}
    for rec in records:
        if rec.split:
            owners["group_key"][rec.group_key].add(rec.split)
            owners["md5"][rec.md5].add(rec.split)
            owners["source_key"][rec.source_key].add(rec.split)
    result = {f"shared_{name}": sum(1 for s in mapping.values() if len(s) > 1)
              for name, mapping in owners.items()}
    result["passed"] = all(v == 0 for v in result.values())
    return result


def split_statistics(records: Sequence[ImageRecord], ratios: Sequence[float],
                     names: dict[int, str]) -> dict:
    """Thống kê theo split và theo class (số lượng, %, sai lệch so với mục tiêu)."""
    per_split: dict[str, dict] = {}
    class_counts = {s: np.zeros(NUM_KEEP, dtype=np.int64) for s in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        recs = [r for r in records if r.split == split]
        for rec in recs:
            for box in rec.boxes:
                class_counts[split][box[0]] += 1
        n_neg = sum(1 for r in recs if not r.boxes)
        per_split[split] = {
            "images": len(recs),
            "positives": len(recs) - n_neg,
            "negatives": n_neg,
            "negative_share_pct": round(100.0 * n_neg / max(len(recs), 1), 2),
            "boxes": int(class_counts[split].sum()),
        }

    total_images = sum(v["images"] for v in per_split.values())
    for split, ratio in zip(SPLIT_NAMES, ratios):
        per_split[split]["image_share_pct"] = round(
            100.0 * per_split[split]["images"] / max(total_images, 1), 2)
        per_split[split]["target_pct"] = round(100.0 * ratio, 2)

    per_class: dict[str, dict] = {}
    for cls in KEEP_CLASSES:
        total = int(sum(class_counts[s][cls] for s in SPLIT_NAMES))
        entry: dict = {"name": class_name(names, cls), "total": total}
        deviations = []
        for split, ratio in zip(SPLIT_NAMES, ratios):
            count = int(class_counts[split][cls])
            pct = 100.0 * count / max(total, 1)
            entry[split] = count
            entry[f"{split}_pct"] = round(pct, 2)
            deviations.append(abs(pct - 100.0 * ratio))
        # Class vắng mặt (total = 0) -> không có sai lệch để đo (None).
        entry["max_abs_dev_pct"] = round(max(deviations), 2) if total else None
        per_class[str(cls)] = entry
    present = [e["max_abs_dev_pct"] for e in per_class.values() if e["total"]]
    return {
        "per_split": per_split,
        "per_class": per_class,
        "absent_classes": [int(c) for c, e in per_class.items() if not e["total"]],
        "max_abs_dev_pct_over_classes": max(present) if present else None,
    }


# ============================================================================
# Bước G - Xuất dữ liệu
# ============================================================================
def assign_output_names(records: Sequence[ImageRecord]) -> int:
    """Đặt tên file đầu ra; xử lý trùng tên một cách tất định.

    Trùng tên (so sánh stem không phân biệt hoa thường vì Windows và vì ảnh
    a.jpg / a.png sẽ đụng nhãn a.txt) trong cùng một split -> thêm tiền tố là
    thư mục split gốc; nếu vẫn trùng -> thêm 8 ký tự md5; nếu vẫn trùng (nhiều
    file giống hệt nhau cùng tên) -> thêm số thứ tự. Trả về số file đổi tên.
    """
    renamed = 0
    for split in SPLIT_NAMES:
        recs = [r for r in records if r.split == split]
        stem_count = Counter(r.src.stem.lower() for r in recs)
        used: set[str] = set()
        for rec in recs:  # records đã được sắp xếp tất định
            name, prefix = rec.src.name, f"{rec.orig_split}_{rec.md5[:8]}"
            candidates = [name] if stem_count[rec.src.stem.lower()] == 1 else []
            candidates += [f"{rec.orig_split}_{name}", f"{prefix}_{name}"]
            numbered = (f"{prefix}_{n}_{name}" for n in itertools.count(2))
            rec.out_name = next(c for c in itertools.chain(candidates, numbered)
                                if Path(c).stem.lower() not in used)
            used.add(Path(rec.out_name).stem.lower())
            renamed += rec.out_name != name
    return renamed


def prepare_output_dirs(out_dir: Path) -> None:
    """Xoá và tạo lại CHỈ các thư mục {Train,Val,Test}/{images,labels}."""
    for split in SPLIT_NAMES:
        split_dir = out_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for sub in ("images", "labels"):
            (split_dir / sub).mkdir(parents=True, exist_ok=True)


def export_record(rec: ImageRecord, out_dir: Path) -> None:
    """Copy ảnh (hoặc encode lại nếu JPEG bị cắt cụt) và ghi file nhãn đã làm sạch."""
    img_dst = out_dir / rec.split / "images" / rec.out_name
    lbl_dst = out_dir / rec.split / "labels" / (Path(rec.out_name).stem + ".txt")
    if rec.truncated:
        image, _ = decode_image(np.fromfile(str(rec.src), dtype=np.uint8))
        ok, encoded = cv2.imencode(".jpg", image,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), REPAIR_JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"Không encode lại được ảnh {rec.rel}")
        encoded.tofile(str(img_dst))
    else:
        # copyfile (không phải copy2): chỉ chép NỘI DUNG, không mang theo thuộc
        # tính read-only của file gốc -> lần chạy sau rmtree không bị PermissionError.
        shutil.copyfile(rec.src, img_dst)
    # Ảnh negative -> file nhãn RỖNG (quy ước YOLO cho background).
    text = "".join(format_box(b) + "\n" for b in rec.boxes)
    lbl_dst.write_text(text, encoding="utf-8", newline="\n")


def write_dataset_yaml(out_dir: Path, names: dict[int, str],
                       split_sizes: dict[str, int]) -> Path:
    """Sinh dataset_filtered.yaml (Ultralytics) với 10 class, GIỮ NGUYÊN ID 0..9.

    Ultralytics bắt buộc có ``val``: nếu Val rỗng (vd. --ratios 0.9 0 0.1) thì
    trỏ ``val`` sang Test; ``test`` chỉ được ghi khi Test có ảnh.
    """
    val_split = "Val" if split_sizes.get("Val") or not split_sizes.get("Test") else "Test"
    content: dict = {
        "path": out_dir.resolve().as_posix(),
        "train": "Train/images",
        "val": f"{val_split}/images",
    }
    if split_sizes.get("Test"):
        content["test"] = "Test/images"
    content["nc"] = NUM_KEEP
    content["names"] = {cls: class_name(names, cls) for cls in KEEP_CLASSES}
    path = out_dir / "dataset_filtered.yaml"
    header = ("# Sinh tự động bởi data_engineering/clean_and_filter.py\n"
              "# 10 class giữ NGUYÊN ID gốc 0..9 (9 biển báo + Pedestrian); mọi class "
              "ngoài 0..9 (10..16 và class lạ như 17) đã bị loại, KHÔNG re-index.\n")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)
    return path


def write_negatives_review_sheet(negatives: Sequence[ImageRecord], path: Path,
                                 seed: int) -> Path | None:
    """Ghép ngẫu nhiên (có seed) tối đa REVIEW_SHEET_SIZE negatives được giữ thành
    một ảnh lưới để KIỂM TRA BẰNG MẮT: nếu thấy biển báo/người đi bộ trong ô nào
    thì ảnh đó thực ra chưa được gán nhãn (xem mục 5 của phần Background Negatives).
    Mỗi ô ghi split + lý do vào pool + tên file đầu ra (để tìm lại trong manifest).
    """
    if not negatives:
        return None
    ordered = sorted(negatives, key=lambda r: natural_key(r.rel))
    sample = random.Random(seed).sample(ordered, min(REVIEW_SHEET_SIZE, len(ordered)))
    tile_w, tile_h = REVIEW_TILE_WH
    cols = 8
    rows = math.ceil(len(sample) / cols)
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, rec in enumerate(sample):
        image, _ = decode_image(np.fromfile(str(rec.src), dtype=np.uint8))
        if image is None:
            continue
        scale = min(tile_w / image.shape[1], tile_h / image.shape[0])
        resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)),
                                     max(1, int(image.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
        y0, x0 = (i // cols) * tile_h, (i % cols) * tile_w
        sheet[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        captions = (f"{i} {rec.split} {negative_reason(rec)}", rec.out_name[:34])
        for line_no, text in enumerate(captions):
            org = (x0 + 3, y0 + 14 + 14 * line_no)
            cv2.putText(sheet, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(sheet, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1,
                        cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    encoded.tofile(str(path))
    return path


def write_manifest(records: Sequence[ImageRecord], path: Path) -> Path:
    """Ghi split_manifest.csv: một dòng cho mỗi ảnh được xuất."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["out_relpath", "split", "group_key", "chunk_id",
                         "n_boxes", "is_negative", "source_path"])
        for rec in records:
            writer.writerow([f"{rec.split}/images/{rec.out_name}", rec.split,
                             rec.group_key, rec.chunk_id, len(rec.boxes),
                             int(not rec.boxes), to_display_path(rec.src)])
    return path


# ============================================================================
# In bảng ra console (dễ đưa vào báo cáo)
# ============================================================================
def format_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]

    def fmt(row: Sequence[str]) -> str:
        return " | ".join(c.ljust(w) if i == 0 else c.rjust(w)
                          for i, (c, w) in enumerate(zip(row, widths)))

    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([fmt(cells[0]), sep] + [fmt(r) for r in cells[1:]])


def print_summary(report: dict) -> None:
    img = report["image_cleaning"]
    lab = report["label_cleaning"]
    neg = report["background_negatives"]
    spl = report["split"]
    print("\n================ TÓM TẮT LÀM SẠCH ================")
    print(f"Ảnh tìm thấy           : {report['inputs']['n_images_found']}")
    print(f"Ảnh 0 byte (bỏ)        : {img['zero_byte']}")
    print(f"Ảnh hỏng (bỏ)          : {img['corrupted']}")
    print(f"JPEG cắt cụt (sửa)     : {img['truncated_jpeg_repaired']}")
    print(f"Dòng nhãn đã đọc       : {lab['lines_total']}")
    print("Dòng nhãn bị loại      : " + (", ".join(
        f"{k}={v}" for k, v in lab["drop_reasons"].items()) or "0"))
    print(f"Box được clip vào ảnh  : {lab['clipped_boxes']}")
    print("Box bị lọc theo class  : " + ", ".join(
        f"{k}={v}" for k, v in report["class_filter"]["dropped_boxes_per_class"].items()))
    print(f"Ảnh mơ hồ (bỏ)         : {neg['ambiguous_excluded']}")
    print(f"Positive thiếu nhãn    : {neg['partially_labeled_positives']} (giữ, xem report)")
    print(f"Pool negatives         : {neg['pool_size']} - đáng ngờ {neg['suspect_excluded']} "
          f"= tin cậy {neg['trusted_pool']}")
    print(f"  -> giữ {neg['kept']} ({neg['neg_ratio']:.0%} mỗi split: "
          + ", ".join(f"{s}={n}" for s, n in neg["kept_per_split"].items())
          + f"), bỏ {neg['discarded']}, mất do purge {neg['purged']}")
    print(f"Tỉ lệ negatives cuối   : {neg['final_negative_share_pct']}%")
    print(f"Nhóm / khối            : {spl['n_groups']} nhóm, {spl['n_chunks']} khối "
          f"(chunk_size={report['config']['chunk_size']})")
    print(f"Sai lệch max (greedy -> local search -> sau purge): "
          f"{spl['max_abs_dev_pct_greedy']} -> {spl['max_abs_dev_pct_refined']} -> "
          f"{spl['max_abs_dev_pct_after_purge']} điểm % ({spl['refinement_steps']} bước, "
          f"{spl['purge_aware_rounds']} vòng purge-aware)")
    print(f"Purge gap              : {spl['purged_groups']} nhóm tại "
          f"{spl['n_split_boundaries']} biên ({spl['purged_positive_images']} ảnh positive, "
          f"{spl['purged_negative_pool_images']} ảnh nền)")
    print(f"Kiểm tra rò rỉ         : {spl['leakage_check']}")

    stats = spl["statistics"]
    rows = [[s, v["images"], f"{v['image_share_pct']}%", v["positives"], v["negatives"],
             f"{v['negative_share_pct']}%", v["boxes"]] for s, v in stats["per_split"].items()]
    print("\n" + format_table(["Split", "Images", "Share", "Positives", "Negatives",
                               "Neg%", "Boxes"], rows))
    rows = [[f"{cls} {e['name']}", e["total"], *(e[s] for s in SPLIT_NAMES),
             *(f"{e[s + '_pct']}%" for s in SPLIT_NAMES),
             "-" if e["max_abs_dev_pct"] is None else e["max_abs_dev_pct"]]
            for cls, e in stats["per_class"].items()]
    print("\n" + format_table(["Class", "Total", *SPLIT_NAMES,
                               *(s + "%" for s in SPLIT_NAMES), "MaxDev(pp)"], rows))
    max_dev = stats["max_abs_dev_pct_over_classes"]
    print("\nSai lệch tuyệt đối lớn nhất so với tỉ lệ mục tiêu: "
          + ("- (không có box nào)" if max_dev is None else f"{max_dev} điểm %"))
    for warning in report["warnings"]:
        print(f"CẢNH BÁO: {warning}")


# ============================================================================
# Pipeline chính
# ============================================================================
def run_cleaning_pipeline(cfg: CleanConfig) -> dict:
    """Chạy toàn bộ pipeline A -> G và trả về report (dict, đã ghi ra JSON)."""
    _ensure_utf8_console()
    start = time.perf_counter()
    raw_dir, out_dir = cfg.raw_dir.resolve(), cfg.out_dir.resolve()
    _validate_config(cfg, raw_dir, out_dir)
    # Hai bộ sinh số ngẫu nhiên ĐỘC LẬP (cùng gốc --seed): một cho việc chia tập,
    # một cho việc lấy mẫu negatives -> đổi --neg-ratio không làm đổi phép chia.
    split_rng, negative_rng = random.Random(cfg.seed), random.Random(cfg.seed + 1)
    timings: dict[str, float] = {}

    def lap(stage: str) -> None:
        timings[stage] = round(time.perf_counter() - start - sum(timings.values()), 1)

    # ---- A. Quét dữ liệu ----
    print(f"[A] Quét ảnh trong {raw_dir} ...", flush=True)
    names, names_source = load_class_names(raw_dir)
    records, orphan_labels = discover_dataset(raw_dir)
    if not records:
        raise SystemExit(f"Không tìm thấy ảnh nào trong {raw_dir}")
    print(f"    {len(records)} ảnh, nhãn mồ côi: {orphan_labels}", flush=True)
    lap("A_discover")

    # ---- B + C + D. Kiểm tra ảnh, làm sạch nhãn, lọc lớp ----
    print("[B-D] Kiểm tra ảnh + làm sạch nhãn + lọc class ...", flush=True)
    per_image_stats = run_parallel(lambda r: inspect_record(r, cfg), records,
                                   cfg.workers, "inspect")
    label_stats = LabelStats()
    for st in per_image_stats:
        label_stats.merge(st)
    valid = [r for r in records if r.status == "ok"]
    lap("BCD_inspect_clean_filter")

    # ---- E (phần 1). Xác định pool background negatives ----
    print("[E] Phân loại positive / pool negatives / mơ hồ / đáng ngờ ...", flush=True)
    for rec in valid:
        rec.category = categorize_record(rec)
    positives = [r for r in valid if r.category == "positive"]
    pool = [r for r in valid if r.category == "negative_pool"]
    ambiguous = [r for r in valid if r.category == "ambiguous"]
    trusted_pool = flag_suspect_negatives(pool, positives)
    split_input = sorted(positives + trusted_pool, key=lambda r: natural_key(r.rel))
    if not split_input:
        raise SystemExit("Không còn ảnh hợp lệ nào để chia tập sau khi làm sạch/lọc "
                         "(kiểm tra lại dữ liệu raw và report của bước B-D).")

    # ---- F. Sequence-based split (positives + TOÀN BỘ pool tin cậy) ----
    print("[F] Chia tập theo chuỗi (sequence-based, stratified) ...", flush=True)
    split_info = sequence_split(split_input, cfg, split_rng)

    # ---- E (phần 2). Lấy mẫu --neg-ratio negatives TRONG TỪNG split ----
    kept_negatives = sample_negatives_per_split(trusted_pool, cfg.neg_ratio, negative_rng)
    exported = sorted((r for r in positives + kept_negatives if r.split), key=lambda r: (
        SPLIT_NAMES.index(r.split), r.chunk_id, _group_sort_key(r.group_key),
        natural_key(r.rel)))
    if not exported:
        raise SystemExit("Không còn ảnh nào để xuất sau khi chia tập + lấy mẫu negatives.")
    split_info["leakage_check"] = verify_no_leakage(exported)
    split_info["statistics"] = split_statistics(exported, cfg.ratios, names)
    renamed = assign_output_names(exported)
    warnings = _collect_warnings(cfg, split_info)
    lap("EF_negatives_split")

    # ---- G. Xuất dữ liệu ----
    # --dry-run ghi file với tiền tố "dry_run_" -> không bao giờ ghi đè report /
    # manifest mô tả lần xuất thật đang nằm trong out_dir.
    prefix = "dry_run_" if cfg.dry_run else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = None
    if cfg.dry_run:
        print("[G] --dry-run: bỏ qua copy ảnh, chỉ ghi dry_run_* (report, manifest).",
              flush=True)
    else:
        print(f"[G] Xuất {len(exported)} ảnh ra {out_dir} ...", flush=True)
        prepare_output_dirs(out_dir)
        run_parallel(lambda r: export_record(r, out_dir), exported, cfg.workers, "export")
        split_sizes = {s: v["images"] for s, v in
                       split_info["statistics"]["per_split"].items()}
        yaml_path = write_dataset_yaml(out_dir, names, split_sizes)
    manifest_path = write_manifest(exported, out_dir / f"{prefix}split_manifest.csv")
    review_path = write_negatives_review_sheet(
        kept_negatives, out_dir / f"{prefix}negatives_review.jpg", cfg.seed)
    lap("G_export")

    report = _build_report(
        cfg=cfg, raw_dir=raw_dir, out_dir=out_dir, names=names, names_source=names_source,
        records=records, valid=valid, orphan_labels=orphan_labels, label_stats=label_stats,
        positives=positives, pool=pool, trusted_pool=trusted_pool,
        kept_negatives=kept_negatives, ambiguous=ambiguous,
        exported=exported, split_info=split_info, renamed=renamed, warnings=warnings,
        outputs={"dataset_yaml": yaml_path, "manifest": manifest_path,
                 "negatives_review": review_path},
        runtime=time.perf_counter() - start, timings=timings,
    )
    report_path = out_dir / f"{prefix}cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print_summary(report)
    print(f"\nReport  : {report_path}\nManifest: {manifest_path}")
    for label, path in (("YAML    ", yaml_path), ("Review  ", review_path)):
        if path:
            print(f"{label}: {path}")
    print(f"Thời gian chạy: {report['runtime_sec']} s")
    return report


def _collect_warnings(cfg: CleanConfig, split_info: dict) -> list[str]:
    """Cảnh báo các tình huống không làm hỏng pipeline nhưng cần người dùng xem lại."""
    warnings: list[str] = []
    per_split = split_info["statistics"]["per_split"]
    for split, ratio in zip(SPLIT_NAMES, cfg.ratios):
        if ratio > 0 and per_split[split]["images"] == 0:
            warnings.append(
                f"Split {split} RỖNG dù tỉ lệ mục tiêu {ratio:.0%} (chỉ có "
                f"{split_info['n_chunks']} khối) -> giảm --chunk-size.")
    if per_split["Val"]["images"] == 0:
        warnings.append("Val rỗng -> dataset_filtered.yaml trỏ 'val' sang Test/images."
                        if per_split["Test"]["images"] else
                        "Val và Test đều rỗng -> Ultralytics sẽ không có tập 'val' hợp lệ.")
    largest = split_info["largest_group"]
    if largest["images"] > LARGE_GROUP_FACTOR * cfg.chunk_size:
        warnings.append(
            f"Nhóm '{largest['key']}' có {largest['images']} ảnh (rất lớn) -> có thể "
            "nhiều ảnh khác nhau bị gộp nhầm do tên file chung chung; kiểm tra lại.")
    if not split_info["leakage_check"]["passed"]:
        warnings.append(f"Phát hiện rò rỉ giữa các split: {split_info['leakage_check']}")
    return warnings


def _ensure_utf8_console() -> None:
    """Console/pipe trên Windows có thể dùng cp1252 -> ép UTF-8 để in tiếng Việt."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _validate_config(cfg: CleanConfig, raw_dir: Path, out_dir: Path) -> None:
    if not raw_dir.is_dir():
        raise SystemExit(f"Thư mục raw không tồn tại: {raw_dir}")
    if out_dir == raw_dir or raw_dir in out_dir.parents:
        raise SystemExit("--out không được nằm bên trong --raw (tránh ghi đè dữ liệu gốc).")
    # Chiều ngược lại: --raw nằm trong (hoặc chính là) thư mục sẽ bị XOÁ khi xuất
    # (out/Train, out/Val, out/Test) -> dữ liệu đầu vào sẽ bị rmtree trước khi copy.
    for split in SPLIT_NAMES:
        wiped = out_dir / split
        if raw_dir == wiped or wiped in raw_dir.parents:
            raise SystemExit(f"--raw nằm trong {wiped} - thư mục này bị xoá và tạo lại "
                             "khi xuất; hãy copy dữ liệu sang nơi khác hoặc đổi --out.")
    if len(cfg.ratios) != 3 or any(r < 0 for r in cfg.ratios) \
            or not math.isclose(sum(cfg.ratios), 1.0, abs_tol=1e-6) or cfg.ratios[0] <= 0:
        raise SystemExit("--ratios phải gồm 3 số không âm, tổng = 1, Train > 0.")
    if cfg.chunk_size < 1 or cfg.purge_gap < 0 or not 0 <= cfg.neg_ratio <= 1:
        raise SystemExit("--chunk-size >= 1, --purge-gap >= 0, 0 <= --neg-ratio <= 1.")


def _build_report(*, cfg: CleanConfig, raw_dir: Path, out_dir: Path, names: dict[int, str],
                  names_source: str, records: Sequence[ImageRecord],
                  valid: Sequence[ImageRecord], orphan_labels: int, label_stats: LabelStats,
                  positives: Sequence[ImageRecord], pool: Sequence[ImageRecord],
                  trusted_pool: Sequence[ImageRecord],
                  kept_negatives: Sequence[ImageRecord], ambiguous: Sequence[ImageRecord],
                  exported: Sequence[ImageRecord], split_info: dict, renamed: int,
                  warnings: list[str], outputs: dict[str, Path | None], runtime: float,
                  timings: dict[str, float]) -> dict:
    """Gom mọi bộ đếm thành report JSON (khoá tiếng Anh để dễ xử lý bằng máy)."""
    status = Counter(r.status for r in records)
    md5_members: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in valid:
        md5_members[rec.md5].append(rec)
    dup_sets = [m for m in md5_members.values() if len(m) > 1]
    conflicting = sum(1 for m in dup_sets if len({tuple(r.boxes) for r in m}) > 1)

    source_splits: dict[str, set[str]] = defaultdict(set)
    for rec in valid:
        source_splits[rec.source_key].add(rec.orig_split)

    n_final_neg = sum(1 for r in exported if not r.boxes)
    partial = [r.rel for r in positives if is_partially_labeled(r)]
    kept_per_split = Counter(r.split for r in kept_negatives)
    n_pool_purged = split_info["purged_negative_pool_images"]
    config = {k: (to_display_path(v) if isinstance(v, Path) else v)
              for k, v in asdict(cfg).items()}
    return {
        "config": config,
        "inputs": {
            "raw_dir": to_display_path(raw_dir),
            "class_names_source": names_source,
            "class_names": {str(k): v for k, v in sorted(names.items())},
            "n_images_found": len(records),
            "images_per_original_split": dict(sorted(Counter(r.orig_split for r in records).items())),
            "label_files_paired": sum(1 for r in records if r.label_path is not None),
            "orphan_label_files": orphan_labels,
            "original_split_leakage_source_frames": sum(
                1 for s in source_splits.values() if len(s) > 1),
        },
        "image_cleaning": {
            "zero_byte": status.get("zero_byte", 0),
            "corrupted": status.get("corrupted", 0),
            "corrupted_files": [r.rel for r in records if r.status == "corrupted"],
            "valid_images": len(valid),
            "truncated_jpeg_repaired": sum(1 for r in exported if r.truncated),
            "truncated_jpeg_detected": sum(1 for r in valid if r.truncated),
            "truncated_jpeg_files": [r.rel for r in valid if r.truncated],
            "byte_identical_sets": len(dup_sets),
            "byte_identical_images": sum(len(m) for m in dup_sets),
            "byte_identical_sets_with_conflicting_labels": conflicting,
            "resolutions_top10_WxH": {
                f"{w}x{h}": n for (w, h), n in
                Counter((r.width, r.height) for r in valid).most_common(10)},
        },
        "label_cleaning": {
            "lines_total": label_stats.lines_total,
            "drop_reasons": dict(label_stats.drop_reasons.most_common()),
            "invalid_lines_by_class": {str(k): v for k, v in sorted(
                label_stats.invalid_by_class.items(), key=lambda kv: str(kv[0]))},
            "clipped_boxes": label_stats.clipped,
            "polygon_lines_converted": label_stats.polygon_converted,
            "label_states": dict(Counter(r.label_state for r in valid)),
        },
        "class_filter": {
            "kept_classes": list(KEEP_CLASSES),
            "reindexed": False,
            "kept_boxes_per_class": {
                f"{c} {class_name(names, c)}": label_stats.kept_by_class.get(c, 0)
                for c in KEEP_CLASSES},
            "dropped_boxes_per_class": {
                f"{c} {class_name(names, c)}": n
                for c, n in sorted(label_stats.class_filtered_by_class.items())},
        },
        "background_negatives": {
            "neg_ratio": cfg.neg_ratio,
            "sampling": "after split, per split (independent RNG seed+1)",
            "positive_images": len(positives),
            "partially_labeled_positives": len(partial),
            "partially_labeled_files": partial,
            "ambiguous_excluded": len(ambiguous),
            "ambiguous_files": [r.rel for r in ambiguous],
            "pool_size": len(pool),
            "pool_by_reason": dict(Counter(negative_reason(r) for r in pool)),
            "suspect_excluded": len(pool) - len(trusted_pool),
            "trusted_pool": len(trusted_pool),
            "trusted_pool_by_reason": dict(Counter(negative_reason(r) for r in trusted_pool)),
            "purged": n_pool_purged,
            "kept": len(kept_negatives),
            "kept_per_split": {s: kept_per_split.get(s, 0) for s in SPLIT_NAMES},
            "kept_by_reason": dict(Counter(negative_reason(r) for r in kept_negatives)),
            "discarded": len(trusted_pool) - n_pool_purged - len(kept_negatives),
            "final_negatives": n_final_neg,
            "final_negative_share_pct": round(100.0 * n_final_neg / max(len(exported), 1), 2),
        },
        "split": {
            "method": "sequence-based grouped chunks + iterative stratification + purge gap",
            "chunk_size": cfg.chunk_size,
            "purge_gap": cfg.purge_gap,
            "target_ratios": dict(zip(SPLIT_NAMES, cfg.ratios)),
            **split_info,
        },
        "export": {
            "out_dir": to_display_path(out_dir),
            "dry_run": cfg.dry_run,
            "images_exported": 0 if cfg.dry_run else len(exported),
            "renamed_for_collision": renamed,
            **{name: to_display_path(path) if path else None
               for name, path in outputs.items()},
        },
        "warnings": warnings,
        "runtime_sec": round(runtime, 1),
        "timings_sec": timings,
    }


# ============================================================================
# CLI
# ============================================================================
def parse_args(argv: Sequence[str] | None = None) -> CleanConfig:
    p = argparse.ArgumentParser(
        description="Làm sạch, lọc class 0..9 (không re-index), chọn background "
                    "negatives và chia Train/Val/Test theo chuỗi (chống data leakage).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = CleanConfig()
    p.add_argument("--raw", type=Path, default=d.raw_dir, help="Thư mục dữ liệu gốc (quét đệ quy)")
    p.add_argument("--out", type=Path, default=d.out_dir, help="Thư mục dữ liệu đầu ra")
    p.add_argument("--ratios", type=float, nargs=3, default=list(d.ratios),
                   metavar=("TRAIN", "VAL", "TEST"), help="Tỉ lệ chia tập")
    p.add_argument("--chunk-size", type=int, default=d.chunk_size,
                   help="Số nhóm frame liên tiếp trong một khối")
    p.add_argument("--purge-gap", type=int, default=d.purge_gap,
                   help="Số nhóm bỏ đi ở mỗi biên giữa 2 split")
    p.add_argument("--neg-ratio", type=float, default=d.neg_ratio,
                   help="Tỉ lệ ảnh nền (negatives) được giữ lại từ pool (trong từng split)")
    p.add_argument("--seed", type=int, default=d.seed, help="Seed cho mọi lựa chọn ngẫu nhiên")
    p.add_argument("--clip-tol", type=float, default=d.clip_tol,
                   help="Dung sai vượt biên (tỉ lệ W/H) được clip thay vì loại")
    p.add_argument("--min-box-px", type=float, default=d.min_box_px,
                   help="Cạnh box tối thiểu (pixel) sau khi clip")
    p.add_argument("--max-aspect", type=float, default=d.max_aspect,
                   help="Tỉ lệ cạnh tối đa max(w/h, h/w)")
    p.add_argument("--workers", type=int, default=d.workers, help="Số thread IO/decode")
    p.add_argument("--dry-run", action="store_true",
                   help="Không copy ảnh; chỉ ghi dry_run_cleaning_report.json + "
                        "dry_run_split_manifest.csv (không ghi đè file của lần xuất thật)")
    a = p.parse_args(argv)
    return CleanConfig(raw_dir=a.raw, out_dir=a.out, ratios=tuple(a.ratios),
                       chunk_size=a.chunk_size, purge_gap=a.purge_gap,
                       neg_ratio=a.neg_ratio, seed=a.seed, clip_tol=a.clip_tol,
                       min_box_px=a.min_box_px, max_aspect=a.max_aspect,
                       workers=a.workers, dry_run=a.dry_run)


def main(argv: Sequence[str] | None = None) -> None:
    _ensure_utf8_console()  # trước parse_args để cả --help (tiếng Việt) in được
    run_cleaning_pipeline(parse_args(argv))


if __name__ == "__main__":
    main()
