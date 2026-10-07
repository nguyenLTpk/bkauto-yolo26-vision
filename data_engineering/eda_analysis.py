"""
eda_analysis.py - Khám phá dữ liệu (Exploratory Data Analysis) cho bộ dữ liệu YOLO thô.

Mục đích
--------
Script này là bước đầu tiên của Giai đoạn 1 (Data Engineering & Preprocessing) trong dự án
Edge AI Object Detection cho xe RC. Nó quét đệ quy thư mục ``data/raw/`` và trả lời các câu hỏi:

1. Dữ liệu ở định dạng nhãn nào (YOLO .txt / COCO .json / Pascal VOC .xml)?
2. Có bao nhiêu ảnh, bao nhiêu file nhãn, bao nhiêu ảnh hỏng / 0 byte / thiếu nhãn / nhãn mồ côi?
3. Phân bố độ phân giải ảnh ra sao?
4. Mỗi class (xem TOÀN BỘ class gốc 0..16, kể cả class lạ như 17 không khai báo trong yaml)
   có bao nhiêu bounding box, xuất hiện trong bao nhiêu ảnh, kích thước box (small/medium/large)?
5. Có dòng nhãn lỗi nào không (sai số trường, không phải số, toạ độ ngoài [0, 1], w/h <= 0, ...)?
6. Có rò rỉ dữ liệu (data leakage) giữa các split gốc của Roboflow không? (cùng một frame gốc bị
   nhân bản bằng augmentation rồi rơi vào cả train lẫn valid/test) -> lý do để chia lại tập theo
   chuỗi (sequence-based split) ở script ``clean_and_filter.py``.
7. "Target view": nếu CHỈ GIỮ class 0..9 (9 biển báo + người đi bộ, KHÔNG đánh lại ID) và bỏ các
   class >= 10 thì còn bao nhiêu box / ảnh, bao nhiêu ảnh sẽ trở thành background.

Kết quả
-------
Tất cả được ghi vào thư mục ``--out`` (mặc định ``reports/eda/``):

* ``eda_summary.json``            - toàn bộ số liệu (dùng cho báo cáo LaTeX / so sánh về sau).
* ``class_distribution.csv``      - bảng class (dễ chuyển thành bảng LaTeX).
* ``class_distribution.png``      - biểu đồ số box theo class (class giữ lại / bị loại khác màu).
* ``resolution_distribution.png`` - top-N độ phân giải.
* ``boxes_per_image.png``         - histogram số box mỗi ảnh.
* ``bbox_size_distribution.png``  - tỷ lệ box small/medium/large theo class (chuẩn COCO).
* ``copies_per_stem.png``         - số bản sao Roboflow trên mỗi frame gốc (bằng chứng leakage).

Biểu đồ dùng matplotlib nếu có (backend Agg, dpi 200); nếu không import được matplotlib
(hoặc truyền ``--no-matplotlib``) thì tự vẽ bằng OpenCV + NumPy thuần.

Cách dùng
---------
    python data_engineering/eda_analysis.py
    python data_engineering/eda_analysis.py --raw-dir data/raw --out reports/eda
    python data_engineering/eda_analysis.py --no-verify-decode      # nhanh hơn, không giải mã ảnh
    python data_engineering/eda_analysis.py --no-matplotlib --top-n 15

``--verify-decode`` (mặc định BẬT) giải mã đầy đủ mọi ảnh bằng OpenCV để phát hiện ảnh hỏng và
tính md5 (tìm ảnh trùng nội dung giữa các split); trên ~24k ảnh mất khoảng 1 phút.
``--no-verify-decode`` chỉ đọc header nên nhanh hơn, nhưng số ảnh hỏng chỉ là cận dưới.

Hoặc dùng như thư viện:
    from eda_analysis import run_eda
    summary = run_eda(Path("data/raw"), Path("reports/eda"))
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, NamedTuple, Sequence

import cv2
import numpy as np
import yaml

# ============================================================================
# Hằng số & cấu hình mặc định
# ============================================================================

# Gốc dự án = thư mục cha của data_engineering/ -> script chạy được từ bất kỳ đâu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "eda"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YAML_NAMES = ("data.yaml", "dataset.yaml")

# Yêu cầu đã chốt với giảng viên: GIỮ NGUYÊN ID 0..9, bỏ mọi class khác (10..16, 17, ...).
KEPT_CLASS_IDS = frozenset(range(10))

# Tên class dự phòng khi không tìm thấy data.yaml (lấy từ data.yaml của bản export Roboflow).
DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "Crosswalk sign", 1: "Highway entrance sign", 2: "Highway exit sign",
    3: "No-entry road sign", 4: "One-way road sign", 5: "Parking sign",
    6: "Priority sign", 7: "Roundabout sign", 8: "Stop sign", 9: "Pedestrian",
    10: "Car", 11: "Green Light", 12: "Red Light", 13: "Yellow Light",
    14: "Closed road stand", 15: "Parking spot", 16: "Stop line",
}

# Ngưỡng kích thước box theo chuẩn COCO (tính bằng diện tích pixel).
SMALL_AREA_PX = 32 ** 2
MEDIUM_AREA_PX = 96 ** 2
SIZE_BUCKETS = ("small", "medium", "large")

EXTREME_ASPECT = 20.0      # box dài/hẹp bất thường (cạnh dài / cạnh ngắn > 20)
COORD_EPS = 1e-6           # dung sai làm tròn khi kiểm tra toạ độ chuẩn hoá trong [0, 1]
# Quy tắc "box dùng được" của target view - cùng giá trị mặc định với clean_and_filter.py:
# box lệch biên <= 1% kích thước ảnh được clip lại, box sau clip có cạnh < 2 px hoặc tỷ lệ
# cạnh > EXTREME_ASPECT thì bị loại.
TARGET_CLIP_TOL = 0.01
TARGET_MIN_BOX_PX = 2.0
DENSITY_CAP = 10           # histogram số box/ảnh: 0, 1, ..., 9, ">=10"
HEADER_BYTES = 64 * 1024   # số byte đầu file đủ để đọc header JPEG/PNG trong hầu hết trường hợp
MAX_EXAMPLES = 5           # số ví dụ lỗi lưu vào JSON cho mỗi loại lỗi
SPLIT_ORDER = ("train", "valid", "val", "test")
TAIL_BYTES = 4096          # số byte cuối file đọc để kiểm tra marker EOI của JPEG

# Các file .txt KHÔNG phải nhãn YOLO dù nằm cạnh ảnh (danh sách tên class, README, danh sách
# ảnh theo split của Roboflow/Darknet). Chỉ áp dụng cho file .txt không ghép được với ảnh nào.
_NON_LABEL_TXT = re.compile(r"(?:classes|obj|readme.*|license.*|train|valid|val|test)\.txt$",
                            re.IGNORECASE)

# Toàn bộ file mà script ghi ra --out (xoá trước mỗi lần chạy để không còn kết quả cũ lẫn vào).
CHART_FILES = ("class_distribution.png", "resolution_distribution.png", "boxes_per_image.png",
               "bbox_size_distribution.png", "copies_per_stem.png")
OUTPUT_FILES = (*CHART_FILES, "class_distribution.csv", "eda_summary.json")

# Bảng màu biểu đồ (đã kiểm tra tương phản trên nền trắng, an toàn cho người mù màu).
COLOR_KEPT = "#2a78d6"        # class 0..9 được giữ lại
COLOR_DROPPED = "#9c9a92"     # class >= 10 bị loại
COLOR_UNDECLARED = "#eb6834"  # class có trong nhãn nhưng không khai báo trong yaml
COLOR_SINGLE = "#2a78d6"
COLOR_SIZES = ("#86b6ef", "#2a78d6", "#104281")  # small / medium / large (thang tuần tự 1 màu)
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_2 = "#52514e"
COLOR_GRID = "#e6e5e0"

# Regex phục vụ phân tích leakage / chuỗi frame.
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-fA-F]{8,}$")               # hậu tố Roboflow ".rf.<hash>"
_EXT_TOKEN = re.compile(r"_(?:jpe?g|png|bmp|webp|tiff?)$", re.IGNORECASE)  # "_jpg" Roboflow thêm
_COPY_SUFFIXES = (
    re.compile(r"[\s_-]*(?:kopya|copy|kopie|copie)$", re.IGNORECASE),  # "-Kopya", " - Copy"
    re.compile(r"\s*\(\d+\)$"),                  # "(1)"
    re.compile(r"-\d+-$"),                       # "(1)" bị Roboflow đổi thành "-1-"
)
_HEX_TOKEN = re.compile(r"(?<![0-9A-Za-z])(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{16,}(?![0-9A-Za-z])")
_DIGITS = re.compile(r"(?<!mp)\d+")   # giữ nguyên "mp4" để họ tên file dễ đọc
_LAST_DIGITS = re.compile(r"\d+(?=\D*$)")
_NATURAL_SPLIT = re.compile(r"(\d+)")

_JPEG_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


# ============================================================================
# Cấu trúc dữ liệu
# ============================================================================

class Box(NamedTuple):
    """Một bounding box YOLO đã parse được (toạ độ chuẩn hoá)."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    valid: bool  # True nếu hình học hợp lệ: w, h > 0 và box nằm trọn trong ảnh


class LabelIssue(NamedTuple):
    """Một dòng nhãn có vấn đề (lỗi hoặc cảnh báo)."""

    kind: str
    line_no: int
    text: str
    # Với dòng hỏng (không thành box): class đọc được từ trường đầu tiên, None nếu không rõ.
    class_id: int | None = None


@dataclass
class ImageRecord:
    """Kết quả kiểm tra một ảnh + file nhãn đi kèm."""

    path: Path
    split: str
    label_path: Path | None
    file_size: int = 0
    width: int | None = None
    height: int | None = None
    header_parsed: bool = False
    decodable: bool | None = None       # None = không kiểm tra giải mã (--no-verify-decode)
    exif_rotated: bool = False          # kích thước sau giải mã bị hoán đổi so với header
    jpeg_missing_eoi: bool = False      # JPEG thiếu marker kết thúc (có thể bị cắt cụt)
    label_empty: bool = False           # có file nhãn nhưng không có dòng nào
    md5: str | None = None              # hash nội dung (chỉ tính khi verify_decode)
    read_error: str | None = None       # lỗi IO khi đọc ảnh/nhãn (file bị khoá, mất quyền...)
    boxes: list[Box] = field(default_factory=list)
    issues: list[LabelIssue] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Ảnh dùng được: đọc được, không 0 byte, không hỏng, biết kích thước."""
        return (self.read_error is None and self.file_size > 0
                and self.decodable is not False and self.width is not None)


@dataclass
class DatasetScan:
    """Kết quả quét thư mục thô (trước khi đọc nội dung ảnh)."""

    images: list[Path]
    label_for_image: dict[Path, Path | None]
    orphan_labels: list[Path]
    format_info: dict[str, Any]
    yaml_files: list[Path] = field(default_factory=list)       # data.yaml / dataset.yaml
    classes_txt_files: list[Path] = field(default_factory=list)  # classes.txt (LabelImg)


@dataclass
class BarChart:
    """Mô tả trung lập một biểu đồ cột, dùng chung cho cả renderer matplotlib và OpenCV."""

    filename: str
    title: str
    subtitle: str
    xlabel: str
    ylabel: str
    categories: list[str]
    series: list[tuple[str, list[float], str]]          # (tên, giá trị, màu hex)
    bar_colors: list[str] | None = None                 # màu riêng từng cột (khi 1 series)
    legend: list[tuple[str, str]] = field(default_factory=list)  # (nhãn, màu)
    value_labels: list[str] | None = None               # nhãn trên đỉnh mỗi cột
    stacked_percent: bool = False                       # cột chồng 100%


# ============================================================================
# Tiện ích chung
# ============================================================================

def natural_key(text: str) -> list[Any]:
    """
    Khoá sắp xếp tự nhiên (alpha-numeric): "frame_2" đứng trước "frame_10".

    ``re.split`` với nhóm bắt luôn trả về [chữ, số, chữ, số, ..., chữ] nên phần tử ở vị trí LẺ
    chắc chắn là dãy số -> đổi sang int theo vị trí (không dùng ``str.isdigit`` vì nó nhận cả
    ký tự như "²" mà ``int()`` không đổi được).
    """
    return [int(tok) if i % 2 else tok.lower()
            for i, tok in enumerate(_NATURAL_SPLIT.split(text))]


def split_sort_key(split: str) -> tuple[int, list[Any]]:
    """Sắp xếp tên split theo thứ tự quen thuộc train -> valid/val -> test -> còn lại."""
    low = split.lower()
    rank = SPLIT_ORDER.index(low) if low in SPLIT_ORDER else len(SPLIT_ORDER)
    return rank, natural_key(split)


def pct(part: float, total: float) -> float:
    """Phần trăm làm tròn 2 chữ số, an toàn khi total = 0."""
    return round(100.0 * part / total, 2) if total else 0.0


def fmt_int(value: float) -> str:
    """Định dạng số nguyên có dấu phân cách hàng nghìn."""
    return f"{int(round(value)):,}"


def relpath(path: Path, root: Path) -> str:
    """Đường dẫn tương đối dạng posix (gọn cho báo cáo); trả về tuyệt đối nếu không cùng gốc."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def configure_console() -> None:
    """Ép stdout sang UTF-8 để in tiếng Việt không lỗi trên console Windows / khi redirect."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ============================================================================
# Đọc tên class từ yaml
# ============================================================================

def _names_from_yaml(yaml_path: Path) -> dict[int, str] | None:
    """Đọc ``names`` (list hoặc dict ``{id: tên}``) của một file yaml; None nếu không hợp lệ."""
    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="replace"))
        names = content.get("names") if isinstance(content, dict) else None
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        if isinstance(names, dict):
            # Khoá phải là số nguyên (0 hoặc "0"); yaml dạng {tên: id} của công cụ khác -> bỏ qua.
            return {int(k): str(v) for k, v in names.items()}
    except (yaml.YAMLError, OSError, ValueError, TypeError):
        pass
    return None


def _names_from_classes_txt(txt_path: Path) -> dict[int, str] | None:
    """Đọc classes.txt kiểu LabelImg (dòng thứ i = tên class i); None nếu rỗng hoặc lỗi đọc."""
    try:
        text = txt_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    names = [line.strip() for line in text.splitlines() if line.strip()]
    return dict(enumerate(names)) or None


def load_class_names(raw_dir: Path, yaml_files: Sequence[Path] | None = None,
                     classes_txt_files: Sequence[Path] = (),
                     ) -> tuple[dict[int, str], set[int], Path | None]:
    """
    Đọc tên class: ưu tiên ``names`` trong data.yaml / dataset.yaml (file nông nhất trước),
    sau đó tới classes.txt (LabelImg), cuối cùng là DEFAULT_CLASS_NAMES.

    ``yaml_files`` là danh sách yaml đã thu thập khi quét (``scan_dataset``) để khỏi duyệt cây
    thư mục lần nữa; nếu None thì tự tìm đệ quy trong ``raw_dir``. File yaml hỏng / ``names``
    không hợp lệ bị bỏ qua (không làm dừng chương trình).
    Trả về (bảng tên, tập ID được khai báo, file nguồn - None nếu dùng tên mặc định).
    """
    if yaml_files is None:
        yaml_files = [p for name in YAML_NAMES for p in raw_dir.rglob(name) if p.is_file()]

    def shallow_first(paths: Iterable[Path]) -> list[Path]:
        return sorted(paths, key=lambda p: (len(p.parts), natural_key(p.as_posix())))

    for source in shallow_first(yaml_files):
        parsed = _names_from_yaml(source)
        if parsed:
            return parsed, set(parsed), source
    for source in shallow_first(classes_txt_files):
        parsed = _names_from_classes_txt(source)
        if parsed:
            return parsed, set(parsed), source
    return dict(DEFAULT_CLASS_NAMES), set(DEFAULT_CLASS_NAMES), None


def class_name(class_id: int, names: dict[int, str]) -> str:
    """Tên class; ID không khai báo (ví dụ 17) hiển thị là ``class_17``."""
    return names.get(class_id, f"class_{class_id}")


# ============================================================================
# Quét thư mục & nhận diện định dạng nhãn
# ============================================================================

def yolo_label_path(image_path: Path, txt_files: set[Path]) -> Path | None:
    """
    Ghép ảnh -> file nhãn theo quy ước YOLO/Ultralytics:
    thay thành phần ``images`` CUỐI CÙNG trong đường dẫn bằng ``labels`` và đổi đuôi thành .txt.
    Dự phòng: file .txt cùng thư mục với ảnh. Tra cứu trong tập file đã quét (không gọi stat()).
    """
    parts = list(image_path.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            if candidate in txt_files:
                return candidate
            break
    fallback = image_path.with_suffix(".txt")
    return fallback if fallback in txt_files else None


def _is_coco_json(path: Path) -> bool:
    """File .json là COCO nếu có đủ khoá images / annotations / categories."""
    try:
        if path.stat().st_size > 512 * 1024 * 1024:
            return False
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys()


def _is_voc_xml(path: Path) -> bool:
    """File .xml là Pascal VOC nếu chứa thẻ gốc <annotation>."""
    try:
        with path.open("rb") as handle:
            head = handle.read(4096).decode("utf-8", errors="replace")
    except OSError:
        return False
    return "<annotation" in head


def _walk_files(root: Path, skip_dir: Path | None = None) -> Iterable[Path]:
    """
    Liệt kê đệ quy mọi file (os.walk nhanh hơn rglob + is_file vì không gọi stat từng file).
    ``skip_dir`` (thư mục --out khi nó nằm bên trong raw_dir) bị bỏ qua để biểu đồ PNG của lần
    chạy trước không bị đếm nhầm thành ảnh dữ liệu.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        if skip_dir is not None:
            dirnames[:] = [d for d in dirnames if base / d != skip_dir]
        dirnames.sort(key=natural_key)
        for name in filenames:
            yield base / name


def _is_yolo_label_candidate(txt: Path, raw_dir: Path, claimed: set[Path],
                             image_dirs: set[Path]) -> bool:
    """
    File .txt được coi là "nhãn YOLO" nếu đã ghép được với một ảnh, hoặc nằm cạnh ảnh / trong
    thư mục ``labels/`` (xét đường dẫn TƯƠNG ĐỐI so với raw_dir, bỏ qua các thư mục cha bên
    ngoài). Loại trừ file danh sách / tên class: train.txt, classes.txt, README.txt ...
    """
    if txt in claimed:
        return True
    if _NON_LABEL_TXT.match(txt.name):
        return False
    in_labels_dir = any(part.lower() == "labels" for part in txt.relative_to(raw_dir).parts[:-1])
    return in_labels_dir or txt.parent in image_dirs


def scan_dataset(raw_dir: Path, skip_dir: Path | None = None) -> DatasetScan:
    """
    Quét đệ quy raw_dir một lần: liệt kê ảnh, ghép nhãn YOLO, tìm nhãn mồ côi, thu thập file
    yaml / classes.txt và nhận diện định dạng nhãn (YOLO .txt / COCO .json / Pascal VOC .xml).
    """
    images: list[Path] = []
    txt_files: set[Path] = set()
    json_files: list[Path] = []
    xml_files: list[Path] = []
    yaml_files: list[Path] = []
    for path in _walk_files(raw_dir, skip_dir):
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            images.append(path)
        elif suffix == ".txt":
            txt_files.add(path)
        elif suffix == ".json":
            json_files.append(path)
        elif suffix == ".xml":
            xml_files.append(path)
        elif path.name.lower() in YAML_NAMES:
            yaml_files.append(path)
    images.sort(key=lambda p: natural_key(relpath(p, raw_dir)))

    label_for_image = {img: yolo_label_path(img, txt_files) for img in images}
    claimed = {lbl for lbl in label_for_image.values() if lbl is not None}
    image_dirs = {img.parent for img in images}
    yolo_candidates = {t for t in txt_files
                       if _is_yolo_label_candidate(t, raw_dir, claimed, image_dirs)}
    orphan_labels = sorted(yolo_candidates - claimed, key=lambda p: natural_key(p.as_posix()))
    classes_txt = [t for t in txt_files - claimed if t.name.lower() == "classes.txt"]

    coco_files = [p for p in json_files if _is_coco_json(p)]
    voc_files = [p for p in xml_files if _is_voc_xml(p)]
    counts = {"yolo_txt": len(yolo_candidates), "coco_json": len(coco_files),
              "voc_xml": len(voc_files)}
    if counts["yolo_txt"]:
        detected = "yolo"
    elif counts["coco_json"]:
        detected = "coco"
    elif counts["voc_xml"]:
        detected = "voc"
    else:
        # Không có file nhãn nào. Nếu có ảnh, run_eda vẫn phân tích như YOLO "0 nhãn"
        # (bộ toàn ảnh nền / chưa gán nhãn): ảnh 0 byte, ảnh hỏng, độ phân giải, leakage...
        detected = "none"
    format_info = {
        "detected": detected,
        "file_counts": counts,
        "other_txt_files": sorted(relpath(p, raw_dir) for p in txt_files - yolo_candidates),
        "coco_files": [relpath(p, raw_dir) for p in coco_files[:MAX_EXAMPLES]],
    }
    return DatasetScan(images, label_for_image, orphan_labels, format_info,
                       yaml_files=yaml_files, classes_txt_files=classes_txt)


def split_of(image_path: Path) -> str:
    """
    Tên split gốc của ảnh: thư mục cha (``images/train/x.jpg`` -> ``train``);
    nếu thư mục cha tên ``images`` thì lấy cấp trên nữa (``train/images/x.jpg`` -> ``train``).
    """
    parent = image_path.parent
    if parent.name.lower() == "images" and parent.parent != parent:
        return parent.parent.name
    return parent.name


# ============================================================================
# Đọc header ảnh (nhanh, chỉ dùng stdlib struct)
# ============================================================================

def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Đọc (W, H) từ marker SOF của JPEG. Trả None nếu không tìm thấy trong ``data``."""
    if not data.startswith(b"\xff\xd8"):
        return None
    pos, size = 2, len(data)
    while pos + 4 <= size:
        if data[pos] != 0xFF:          # mất đồng bộ -> dò tiếp tới byte 0xFF kế tiếp
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == 0xFF:             # byte đệm (fill byte)
            pos += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # marker không có độ dài
            pos += 2
            continue
        if marker in (0xD9, 0xDA):     # tới EOI/SOS mà chưa thấy SOF -> bỏ cuộc
            return None
        if marker in _JPEG_SOF_MARKERS:
            if pos + 9 > size:
                return None
            height, width = struct.unpack(">HH", data[pos + 5:pos + 9])
            return (width, height) if width and height else None
        segment_length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        pos += 2 + segment_length
    return None


def parse_image_size(data: bytes) -> tuple[int, int] | None:
    """Đọc (W, H) từ header JPEG (SOF), PNG (IHDR) hoặc BMP; định dạng khác trả None."""
    if data.startswith(b"\xff\xd8"):
        return _jpeg_size(data)
    if data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR" and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return (width, height) if width and height else None
    if data.startswith(b"BM") and len(data) >= 26:
        width, height = struct.unpack("<ii", data[18:26])
        return (abs(width), abs(height)) if width and height else None
    return None


# ============================================================================
# Parse nhãn YOLO
# ============================================================================

def _box_inside_image(cx: float, cy: float, w: float, h: float) -> bool:
    """Box (chuẩn hoá) nằm trọn trong ảnh: tâm và 4 cạnh đều trong [0, 1] (có dung sai)."""
    lo, hi = -COORD_EPS, 1.0 + COORD_EPS
    return all(lo <= v <= hi for v in (cx, cy, w, h)) and (
        cx - w / 2 >= lo and cx + w / 2 <= hi and cy - h / 2 >= lo and cy + h / 2 <= hi
    )


def parse_yolo_label(text: str, width: int | None,
                     height: int | None) -> tuple[list[Box], list[LabelIssue], bool]:
    """
    Parse nội dung một file nhãn YOLO ``class cx cy w h`` (toạ độ chuẩn hoá).

    Trả về (danh sách box, danh sách vấn đề, file có rỗng không). Các loại vấn đề:
    - wrong_field_count / non_numeric / invalid_class_id: dòng hỏng, KHÔNG tính vào box.
    - non_positive_size / coords_out_of_range: box được tính (raw view) nhưng ``valid=False``.
    - extreme_aspect: chỉ là cảnh báo (cạnh dài / cạnh ngắn > 20 tính theo pixel).
    - duplicate_line: dòng trùng lặp hoàn toàn trong cùng một file (vẫn được tính).
    """
    boxes: list[Box] = []
    issues: list[LabelIssue] = []
    seen: set[str] = set()
    non_blank = 0
    for line_no, raw in enumerate(text.lstrip("﻿").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        non_blank += 1
        normalized = " ".join(line.split())
        if normalized in seen:
            issues.append(LabelIssue("duplicate_line", line_no, line))
        seen.add(normalized)

        fields = line.split()
        line_class = _class_token(fields[0])
        if len(fields) != 5:
            issues.append(LabelIssue("wrong_field_count", line_no, line, line_class))
            continue
        try:
            values = [float(v) for v in fields]
        except ValueError:
            issues.append(LabelIssue("non_numeric", line_no, line, line_class))
            continue
        if not all(math.isfinite(v) for v in values):
            issues.append(LabelIssue("non_numeric", line_no, line, line_class))
            continue
        cls_value, cx, cy, w, h = values
        if line_class is None:
            issues.append(LabelIssue("invalid_class_id", line_no, line))
            continue

        valid = True
        if w <= 0 or h <= 0:
            issues.append(LabelIssue("non_positive_size", line_no, line))
            valid = False
        if not _box_inside_image(cx, cy, w, h):
            issues.append(LabelIssue("coords_out_of_range", line_no, line))
            valid = False
        if valid:
            pw, ph = (w * width, h * height) if width and height else (w, h)
            if max(pw / ph, ph / pw) > EXTREME_ASPECT:
                issues.append(LabelIssue("extreme_aspect", line_no, line))
        boxes.append(Box(line_class, cx, cy, w, h, valid))
    return boxes, issues, non_blank == 0


def _class_token(token: str) -> int | None:
    """Đọc class id từ trường đầu của dòng nhãn ("3", "3.0"); None nếu không phải số nguyên >= 0."""
    try:
        value = float(token)
    except ValueError:
        return None
    return int(value) if math.isfinite(value) and value >= 0 and value.is_integer() else None


def target_box_usable(box: Box, width: int, height: int) -> bool:
    """
    Box có dùng được cho tập huấn luyện không - xấp xỉ quy tắc mặc định của clean_and_filter.py:
    w, h > 0; tâm nằm trong ảnh; các cạnh lệch biên tối đa 1% kích thước ảnh (phần lệch được
    clip); sau khi clip mỗi cạnh >= 2 px và tỷ lệ cạnh dài/ngắn <= 20.
    """
    bw, bh = box.w * width, box.h * height
    pcx, pcy = box.cx * width, box.cy * height
    if bw <= 0 or bh <= 0 or not (0 <= pcx <= width and 0 <= pcy <= height):
        return False
    x1, y1, x2, y2 = pcx - bw / 2, pcy - bh / 2, pcx + bw / 2, pcy + bh / 2
    tol_x, tol_y = TARGET_CLIP_TOL * width, TARGET_CLIP_TOL * height
    if x1 < -tol_x or y1 < -tol_y or x2 > width + tol_x or y2 > height + tol_y:
        return False
    bw = min(float(width), x2) - max(0.0, x1)
    bh = min(float(height), y2) - max(0.0, y1)
    if bw < TARGET_MIN_BOX_PX or bh < TARGET_MIN_BOX_PX:
        return False
    return max(bw / bh, bh / bw) <= EXTREME_ASPECT


# ============================================================================
# Kiểm tra từng ảnh (chạy song song bằng thread)
# ============================================================================

def _read_head_and_tail(path: Path, file_size: int) -> tuple[bytes, bytes]:
    """Đọc 64 KB đầu và 4 KB cuối file (không giải mã) - đủ cho header & marker EOI."""
    with path.open("rb") as handle:
        head = handle.read(HEADER_BYTES)
        handle.seek(max(0, file_size - TAIL_BYTES))
        tail = handle.read(TAIL_BYTES)
    return head, tail


def _jpeg_ends_with_eoi(tail: bytes) -> bool:
    """
    JPEG kết thúc "sạch" nếu marker EOI (FF D9) nằm trong 64 byte cuối, sau khi bỏ các byte 0
    đệm ở đuôi. Ngược lại: file có thể bị cắt cụt, hoặc có dữ liệu thừa sau EOI (trailer của
    điện thoại, Motion Photo...) - chỉ là cảnh báo, ảnh hỏng thật do bước giải mã quyết định.
    """
    return b"\xff\xd9" in tail.rstrip(b"\x00")[-64:]


def _inspect_image_file(record: ImageRecord, verify_decode: bool) -> None:
    """
    Đọc file ảnh của ``record``: kích thước (header -> fallback giải mã), khả năng giải mã bằng
    OpenCV và md5 (khi ``verify_decode``). Đọc bằng ``Path.read_bytes`` + ``np.frombuffer`` +
    ``cv2.imdecode``: an toàn với đường dẫn Unicode và (khác ``np.fromfile``) báo lỗi OSError
    thật khi file bị khoá thay vì trả về dữ liệu rác.
    """
    image_path = record.path
    record.file_size = image_path.stat().st_size
    if record.file_size == 0:
        record.decodable = False if verify_decode else None
        return

    data = image_path.read_bytes() if verify_decode else None
    if data is not None:
        head, tail = data[:HEADER_BYTES], data[-TAIL_BYTES:]
        record.md5 = hashlib.md5(data).hexdigest()
    else:
        head, tail = _read_head_and_tail(image_path, record.file_size)

    size = parse_image_size(head)
    if size is None and len(head) < record.file_size and head.startswith(b"\xff\xd8"):
        # SOF nằm sau 64 KB đầu (ví dụ EXIF thumbnail lớn) -> đọc cả file rồi thử lại.
        data = image_path.read_bytes() if data is None else data
        size = parse_image_size(data)
    record.header_parsed = size is not None
    if size is not None:
        record.width, record.height = size
    if head.startswith(b"\xff\xd8"):
        record.jpeg_missing_eoi = not _jpeg_ends_with_eoi(tail)

    if verify_decode or size is None:
        data = image_path.read_bytes() if data is None else data
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        record.decodable = image is not None
        if image is not None:
            dec_h, dec_w = image.shape[:2]
            # cv2 tự xoay theo EXIF orientation -> kích thước thực tế mà pipeline nhìn thấy.
            record.exif_rotated = size is not None and (dec_w, dec_h) != size
            record.width, record.height = dec_w, dec_h


def inspect_image(image_path: Path, label_path: Path | None, verify_decode: bool) -> ImageRecord:
    """
    Kiểm tra một ảnh (dung lượng, kích thước, giải mã, md5) rồi parse file nhãn đi kèm.

    Lỗi IO của TỪNG file (file bị antivirus/OneDrive khoá, mất quyền đọc, bị xoá giữa chừng)
    không làm dừng cả lần chạy: ảnh được ghi nhận ``read_error`` và báo cáo riêng ở mục
    "file không đọc được" (khác với ảnh hỏng - ảnh đọc được nhưng cv2 không giải mã được).
    """
    record = ImageRecord(path=image_path, split=split_of(image_path), label_path=label_path)
    try:
        _inspect_image_file(record, verify_decode)
        if label_path is not None:
            text = label_path.read_text(encoding="utf-8", errors="replace")
            record.boxes, record.issues, record.label_empty = parse_yolo_label(
                text, record.width, record.height)
    except OSError as error:
        source = f" ({Path(error.filename).name})" if error.filename else ""
        record.read_error = f"{type(error).__name__}: {error.strerror or error}{source}"
        record.decodable = None
    return record


def inspect_all(scan: DatasetScan, verify_decode: bool, workers: int) -> list[ImageRecord]:
    """Chạy ``inspect_image`` song song (cv2/IO nhả GIL nên thread là đủ) và in tiến độ."""
    total = len(scan.images)
    step = max(1, total // 10)
    records: list[ImageRecord] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(
            lambda img: inspect_image(img, scan.label_for_image[img], verify_decode), scan.images)
        for index, record in enumerate(results, start=1):
            records.append(record)
            if index % step == 0 or index == total:
                elapsed = time.perf_counter() - started
                print(f"  [eda] {index:>7,}/{total:,} ảnh ({pct(index, total):5.1f}%) "
                      f"- {elapsed:6.1f}s", flush=True)
    return records


# ============================================================================
# Phân tích leakage / chuỗi frame (dựa trên tên file)
# ============================================================================

def source_stem(filename: str) -> str:
    """
    Suy ra "frame gốc" của một file bằng cách bỏ các hậu tố do công cụ sinh ra:

    * ``.rf.<32 hex>``  - Roboflow thêm cho MỖI bản augmentation của cùng một frame;
    * hậu tố bản sao    - ``-Kopya`` (tiếng Thổ Nhĩ Kỳ: "bản sao"), `` - Copy``, ``(1)``
      và ``-1-`` (Roboflow đổi ``(1)`` thành ``-1-``).

    Token đuôi gốc (``_jpg``, ``_png``) được giữ lại để không gộp nhầm ``frame_1_jpg`` và
    ``frame_1_png`` (hai nguồn khác nhau).
    Ví dụ: ``yaya3_mp4-12-Kopya_jpg.rf.ab12...`` -> ``yaya3_mp4-12_jpg``.
    """
    stem = _RF_SUFFIX.sub("", Path(filename).stem)
    ext_match = _EXT_TOKEN.search(stem)
    ext_token = ext_match.group(0) if ext_match else ""
    base = stem[: len(stem) - len(ext_token)]
    changed = True
    while changed:
        changed = False
        for pattern in _COPY_SUFFIXES:
            stripped = pattern.sub("", base)
            if stripped and stripped != base:
                base, changed = stripped, True
    return base + ext_token


def filename_family(stem: str) -> str:
    """Họ tên file: thay hash hex dài bằng ``<hash>`` và mọi dãy số bằng ``#``."""
    return _DIGITS.sub("#", _HEX_TOKEN.sub("<hash>", stem))


def sequence_key(stem: str) -> str:
    """
    Khoá "chuỗi video": chỉ thay dãy số CUỐI (thường là chỉ số frame) bằng ``#``.
    Ví dụ ``real_track_5_left_mp4-117_jpg`` -> ``real_track_5_left_mp4-#_jpg``: mọi frame của
    cùng một video có chung khoá, nên ta biết chuỗi nào bị rải ra nhiều split.
    """
    return _LAST_DIGITS.sub("#", _HEX_TOKEN.sub("<hash>", stem))


# ============================================================================
# Tính thống kê
# ============================================================================

def _issue_examples(records: Iterable[ImageRecord], raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Gom các vấn đề nhãn theo loại: số lượng + vài ví dụ ``file:dòng: nội dung``."""
    grouped: dict[str, dict[str, Any]] = {}
    for rec in records:
        for issue in rec.issues:
            entry = grouped.setdefault(issue.kind, {"count": 0, "examples": []})
            entry["count"] += 1
            if len(entry["examples"]) < MAX_EXAMPLES:
                entry["examples"].append(
                    f"{relpath(rec.label_path, raw_dir)}:{issue.line_no}: {issue.text}")
    return dict(sorted(grouped.items()))


def compute_overview(records: list[ImageRecord], scan: DatasetScan, raw_dir: Path,
                     verify_decode: bool) -> dict[str, Any]:
    """Tổng quan: số ảnh, theo split, có/không nhãn, nhãn rỗng/mồ côi, ảnh 0 byte/hỏng."""
    per_split: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        stats = per_split[rec.split]
        stats["images"] += 1
        stats["with_label_file"] += rec.label_path is not None
        stats["without_label_file"] += rec.label_path is None
        stats["empty_label_files"] += rec.label_empty
        stats["boxes"] += len(rec.boxes)
    unreadable = [r for r in records if r.read_error is not None]
    zero_byte = [r for r in records if r.read_error is None and r.file_size == 0]
    corrupted = [r for r in records if r.read_error is None and r.file_size > 0
                 and r.decodable is False]

    def examples(items: Sequence[ImageRecord]) -> list[str]:
        return [relpath(r.path, raw_dir) for r in items[:MAX_EXAMPLES]]

    return {
        "total_images": len(records),
        "total_size_mb": round(sum(r.file_size for r in records) / 2 ** 20, 1),
        "images_per_split": {s: dict(per_split[s]) for s in sorted(per_split, key=split_sort_key)},
        "images_with_label_file": sum(r.label_path is not None for r in records),
        "images_without_label_file": sum(r.label_path is None for r in records),
        "empty_label_files": sum(r.label_empty for r in records),
        "orphan_label_files": {"count": len(scan.orphan_labels),
                               "examples": [relpath(p, raw_dir)
                                            for p in scan.orphan_labels[:MAX_EXAMPLES]]},
        # File không đọc được (lỗi IO: bị khoá, mất quyền...) - KHÔNG phải ảnh hỏng; chạy lại
        # khi file được giải phóng để có số liệu đầy đủ.
        "unreadable_files": {"count": len(unreadable),
                             "examples": [f"{relpath(r.path, raw_dir)} ({r.read_error})"
                                          for r in unreadable[:MAX_EXAMPLES]]},
        "zero_byte_images": {"count": len(zero_byte), "examples": examples(zero_byte)},
        # Khi --no-verify-decode: chỉ những ảnh không đọc được header mới bị giải mã thử,
        # nên con số này là CẬN DƯỚI (fully_verified = False).
        "corrupted_images": {"count": len(corrupted), "examples": examples(corrupted),
                             "fully_verified": verify_decode},
        "decode_verified": verify_decode,
        "header_unparsed_images": sum(r.read_error is None and r.file_size > 0
                                      and not r.header_parsed for r in records),
        "exif_rotated_images": sum(r.exif_rotated for r in records),
        # JPEG không kết thúc bằng EOI (bỏ qua byte 0 đệm): bị cắt cụt HOẶC có dữ liệu thừa
        # sau EOI. Chỉ là cảnh báo - ảnh hỏng thật được đếm ở corrupted_images.
        "jpeg_missing_eoi": sum(r.jpeg_missing_eoi for r in records),
    }


def compute_resolution_stats(records: list[ImageRecord], top_n: int) -> dict[str, Any]:
    """Phân bố độ phân giải (W x H): top-N + nhóm "khác", hướng ảnh ngang/dọc/vuông."""
    sizes = Counter((r.width, r.height) for r in records if r.width is not None)
    known = sum(sizes.values())
    ranked = sorted(sizes.items(), key=lambda kv: (-kv[1], -kv[0][0] * kv[0][1]))
    top = [{"resolution": f"{w}x{h}", "width": w, "height": h, "count": c, "percent": pct(c, known)}
           for (w, h), c in ranked[:top_n]]
    other = sum(c for _, c in ranked[top_n:])
    orientation: Counter = Counter()
    for (w, h), count in sizes.items():
        orientation["landscape" if w > h else "portrait" if h > w else "square"] += count
    return {
        "images_with_known_size": known,
        "unknown_size": len(records) - known,
        "unique_resolutions": len(sizes),
        "top": top,
        "other": {"resolutions": max(0, len(ranked) - top_n), "count": other,
                  "percent": pct(other, known)},
        "orientation": dict(orientation),
    }


def _size_bucket(box: Box, width: int, height: int) -> str:
    """Phân loại box theo diện tích pixel chuẩn COCO: small < 32², medium < 96², large."""
    area = box.w * width * box.h * height
    if area < SMALL_AREA_PX:
        return "small"
    return "medium" if area < MEDIUM_AREA_PX else "large"


def compute_class_stats(records: list[ImageRecord], names: dict[int, str],
                        declared: set[int]) -> dict[str, Any]:
    """
    Bảng class trên TOÀN BỘ nhãn gốc (raw view): số box, số ảnh chứa class, % box,
    số box hợp lệ hình học, phân bố small/medium/large và cạnh trung vị (pixel).
    """
    boxes = Counter()
    valid_boxes = Counter()
    images_with = Counter()
    buckets: dict[int, Counter] = defaultdict(Counter)
    sides: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        for cls in {b.class_id for b in rec.boxes}:
            images_with[cls] += 1
        for box in rec.boxes:
            boxes[box.class_id] += 1
            if not box.valid:
                continue
            valid_boxes[box.class_id] += 1
            if rec.width and rec.height:
                buckets[box.class_id][_size_bucket(box, rec.width, rec.height)] += 1
                sides[box.class_id].append(math.sqrt(box.w * rec.width * box.h * rec.height))

    total = sum(boxes.values())
    rows = []
    for cls in sorted(declared | set(boxes)):
        rows.append({
            "id": cls,
            "name": class_name(cls, names),
            "declared": cls in declared,
            "status": "kept" if cls in KEPT_CLASS_IDS else "dropped",
            "boxes": boxes[cls],
            "images": images_with[cls],
            "percent_boxes": pct(boxes[cls], total),
            "valid_boxes": valid_boxes[cls],
            "size": {b: buckets[cls][b] for b in SIZE_BUCKETS},
            "median_side_px": round(median(sides[cls]), 1) if sides[cls] else None,
        })
    overall = Counter()
    for counter in buckets.values():
        overall.update(counter)
    all_sides = [s for values in sides.values() for s in values]
    return {
        "total_boxes": total,
        "classes": rows,
        "undeclared_classes": {cls: boxes[cls] for cls in sorted(set(boxes) - declared)},
        "size_overall": {b: overall[b] for b in SIZE_BUCKETS},
        "size_overall_percent": {b: pct(overall[b], sum(overall.values())) for b in SIZE_BUCKETS},
        "median_side_px": round(median(all_sides), 1) if all_sides else None,
    }


def compute_density_stats(records: list[ImageRecord]) -> dict[str, Any]:
    """Mật độ box mỗi ảnh (mọi class): mean/median/max + histogram 0..9, >=10."""
    counts = [len(r.boxes) for r in records]
    labeled = [c for c in counts if c > 0]
    histogram = Counter(min(c, DENSITY_CAP) for c in counts)
    return {
        "mean_all_images": round(mean(counts), 3) if counts else 0.0,
        "median_all_images": median(counts) if counts else 0,
        "mean_labeled_images": round(mean(labeled), 3) if labeled else 0.0,
        "median_labeled_images": median(labeled) if labeled else 0,
        "max": max(counts, default=0),
        "histogram": {(f">={DENSITY_CAP}" if k == DENSITY_CAP else str(k)): histogram[k]
                      for k in range(DENSITY_CAP + 1)},
    }


def compute_leakage_stats(records: list[ImageRecord], top_n: int) -> dict[str, Any]:
    """
    Phân tích rò rỉ dữ liệu giữa các split gốc dựa trên tên file:

    * stem gốc (``source_stem``): nhiều file Roboflow là bản augment của CÙNG một frame;
      nếu một stem xuất hiện ở >1 split thì mô hình được "nhìn thấy" ảnh test khi train.
    * chuỗi video (``sequence_key``): các frame liên tiếp của cùng video gần như giống hệt nhau;
      chuỗi trải trên nhiều split cũng là một dạng leakage -> cần chia tập theo chuỗi/khối.
    """
    stem_files: Counter = Counter()
    stem_splits: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        stem = source_stem(rec.path.name)
        stem_files[stem] += 1
        stem_splits[stem].add(rec.split)

    copies_hist = Counter(stem_files.values())
    leaking = sorted((s for s, sp in stem_splits.items() if len(sp) > 1), key=natural_key)
    split_combos = Counter(" & ".join(sorted(stem_splits[s], key=split_sort_key)) for s in leaking)

    seq_stems: dict[str, set[str]] = defaultdict(set)
    seq_files: Counter = Counter()
    seq_splits: dict[str, set[str]] = defaultdict(set)
    for stem, n_files in stem_files.items():
        key = sequence_key(stem)
        seq_stems[key].add(stem)
        seq_files[key] += n_files
        seq_splits[key] |= stem_splits[stem]
    multi_split_seqs = [k for k in seq_files if len(seq_splits[k]) > 1]

    family_stems: Counter = Counter()
    family_files: Counter = Counter()
    for stem, n_files in stem_files.items():
        family = filename_family(stem)
        family_stems[family] += 1
        family_files[family] += n_files

    return {
        # Phân tích dựa trên TÊN FILE; trùng lặp nội dung được kiểm bằng md5 ở exact_duplicates.
        "method": "filename-based (source_stem / sequence_key)",
        "total_files": len(records),
        "unique_source_stems": len(stem_files),
        "duplicated_files": len(records) - len(stem_files),
        "copies_per_stem_mean": round(len(records) / len(stem_files), 3) if stem_files else 0.0,
        "copies_per_stem_max": max(stem_files.values(), default=0),
        "copies_per_stem_histogram": {str(k): copies_hist[k] for k in sorted(copies_hist)},
        "stems_in_multiple_splits": len(leaking),
        "files_of_leaking_stems": sum(stem_files[s] for s in leaking),
        "leaking_split_combinations": dict(split_combos.most_common()),
        "leaking_stem_examples": leaking[:10],
        "sequences": len(seq_files),
        "sequences_in_multiple_splits": len(multi_split_seqs),
        "files_in_multi_split_sequences": sum(seq_files[k] for k in multi_split_seqs),
        # sequence_key chỉ thay dãy số CUỐI, nên họ tên chung chung (frame_#_jpg, #_Color_jpg)
        # gộp nhiều video khác nhau vào một khoá -> con số trên là ƯỚC LƯỢNG THÔ (cận trên),
        # không phải số frame rò rỉ chính xác (con số chính xác hơn: stems_in_multiple_splits).
        "sequence_key_is_coarse": True,
        "top_sequences": [
            {"sequence": k, "stems": len(seq_stems[k]), "files": seq_files[k],
             "splits": sorted(seq_splits[k], key=split_sort_key)}
            for k, _ in sorted(seq_files.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        ],
        "top_families": [
            {"family": f, "stems": family_stems[f], "files": c}
            for f, c in sorted(family_files.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        ],
        "exact_duplicates": compute_exact_duplicates(records),
    }


def compute_exact_duplicates(records: list[ImageRecord]) -> dict[str, Any] | None:
    """
    Bổ sung cho phân tích theo tên file: nhóm các ảnh TRÙNG NỘI DUNG từng byte (cùng md5, bỏ
    file 0 byte). Đếm nhóm nằm ở >1 split gốc và trong đó các nhóm mà heuristic tên file có
    thể bỏ sót (file mang ``source_stem`` khác nhau). Chỉ có khi verify_decode (md5 được tính
    từ buffer đã đọc sẵn, gần như không tốn thêm thời gian); ngược lại trả None.
    """
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        if rec.md5 is not None and rec.file_size > 0:
            by_hash[rec.md5].append(rec)
    if not by_hash:
        return None
    groups = [g for g in by_hash.values() if len(g) > 1]
    cross_split = [g for g in groups if len({r.split for r in g}) > 1]
    missed = [g for g in cross_split if len({source_stem(r.path.name) for r in g}) > 1]
    return {
        "groups": len(groups),
        "files_in_groups": sum(len(g) for g in groups),
        "groups_across_splits": len(cross_split),
        "groups_across_splits_with_different_stems": len(missed),
    }


def compute_target_view(records: list[ImageRecord], names: dict[int, str]) -> dict[str, Any]:
    """
    "Target view": mô phỏng việc CHỈ GIỮ class 0..9 (giữ nguyên ID, không re-index) và bỏ
    mọi class khác, với quy tắc box XẤP XỈ clean_and_filter.py (``target_box_usable``: clip
    lệch biên <= 1%, cạnh >= 2 px, tỷ lệ cạnh <= 20). Đây là ước lượng nhanh trước khi làm
    sạch: script làm sạch còn đổi polygon sang bbox, bỏ dòng trùng... nên số liệu thực tế có thể
    lệch nhẹ (con số chính xác nằm trong cleaning_report.json).

    Mọi box gốc được phân vào đúng MỘT nhóm, nên cộng lại bằng tổng box của raw view:
    kept (0..9, dùng được) + rejected_kept (0..9 nhưng box lỗi) + dropped (class >= 10)
    + nằm trên ảnh không dùng được (0 byte / hỏng / không đọc được).

    Ảnh không còn box nào chia làm hai loại:
    * background: không có file nhãn, nhãn rỗng, hoặc chỉ chứa class bị loại -> pool để chọn
      ngẫu nhiên 10% làm background negatives;
    * ambiguous (mơ hồ): ảnh CÓ đối tượng class 0..9 (hoặc dòng nhãn hỏng không rõ class) nhưng
      mọi box đều lỗi -> ảnh vẫn chứa biển báo/người thật, KHÔNG được dùng làm negative.
    """
    kept = Counter()
    kept_images = Counter()
    dropped = Counter()
    reasons = Counter()
    unusable = 0
    boxes_on_unusable = 0
    rejected_kept = 0
    ambiguous = 0
    positives_per_split = Counter()
    kept_per_positive: list[int] = []
    for rec in records:
        if not rec.usable:
            unusable += 1
            boxes_on_unusable += len(rec.boxes)
            continue
        target = [b for b in rec.boxes if b.class_id in KEPT_CLASS_IDS]
        kept_boxes = [b for b in target if target_box_usable(b, rec.width, rec.height)]
        rejected_kept += len(target) - len(kept_boxes)
        for box in rec.boxes:
            if box.class_id not in KEPT_CLASS_IDS:
                dropped[box.class_id] += 1
        broken_target_line = any(
            issue.kind in ("wrong_field_count", "non_numeric", "invalid_class_id")
            and (issue.class_id is None or issue.class_id in KEPT_CLASS_IDS)
            for issue in rec.issues)
        if kept_boxes:
            for box in kept_boxes:
                kept[box.class_id] += 1
            for cls in {b.class_id for b in kept_boxes}:
                kept_images[cls] += 1
            positives_per_split[rec.split] += 1
            kept_per_positive.append(len(kept_boxes))
        elif target or broken_target_line:
            ambiguous += 1
        elif rec.label_path is None:
            reasons["no_label_file"] += 1
        elif rec.label_empty:
            reasons["empty_label_file"] += 1
        else:
            reasons["only_dropped_classes"] += 1

    background = sum(reasons.values())
    kept_counts = [kept[c] for c in sorted(KEPT_CLASS_IDS)]
    total_kept = sum(kept.values())
    return {
        "kept_class_ids": sorted(KEPT_CLASS_IDS),
        "box_rules": {"clip_tolerance": TARGET_CLIP_TOL, "min_box_px": TARGET_MIN_BOX_PX,
                      "max_aspect": EXTREME_ASPECT},
        "unusable_images_excluded": unusable,
        "boxes_on_unusable_images": boxes_on_unusable,
        "rejected_kept_class_boxes": rejected_kept,
        "kept_classes": [
            {"id": c, "name": class_name(c, names), "boxes": kept[c], "images": kept_images[c],
             "percent_boxes": pct(kept[c], total_kept)} for c in sorted(KEPT_CLASS_IDS)
        ],
        "kept_boxes_total": total_kept,
        "dropped_boxes_total": sum(dropped.values()),
        "dropped_boxes_by_class": {f"{c} {class_name(c, names)}": dropped[c]
                                   for c in sorted(dropped)},
        # Tỷ lệ mất cân bằng = class nhiều box nhất / class ít box nhất (None nếu có class 0 box).
        "imbalance_ratio_max_min": (round(max(kept_counts) / min(kept_counts), 3)
                                    if min(kept_counts) else None),
        "positive_images": len(kept_per_positive),
        "positive_images_per_split": {s: positives_per_split[s]
                                      for s in sorted(positives_per_split, key=split_sort_key)},
        "mean_kept_boxes_per_positive_image": (round(mean(kept_per_positive), 3)
                                               if kept_per_positive else 0.0),
        "background_images_after_filter": background,
        "background_reasons": dict(reasons.most_common()),
        "background_kept_if_10_percent": round(background * 0.10),
        "ambiguous_images_excluded": ambiguous,
    }


# ============================================================================
# Vẽ biểu đồ - renderer matplotlib
# ============================================================================

def _load_pyplot() -> Any | None:
    """Import matplotlib.pyplot với backend Agg (không cần màn hình); None nếu không có."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def render_matplotlib(chart: BarChart, out_path: Path, plt: Any) -> None:
    """Vẽ ``BarChart`` bằng matplotlib (phong cách gọn cho báo cáo, dpi 200)."""
    n = len(chart.categories)
    long_labels = max((len(c) for c in chart.categories), default=0) > 6
    per_bar = 0.62 if long_labels else 0.4   # nhãn ngắn -> cột hẹp hơn, hình gọn hơn
    fig, ax = plt.subplots(figsize=(max(7.5, per_bar * n + 2.4), 5.4), dpi=200)
    x = np.arange(n)
    bar_width = 0.72
    if chart.stacked_percent:
        bottom = np.zeros(n)
        totals = np.sum([vals for _, vals, _ in chart.series], axis=0).astype(float)
        totals[totals == 0] = 1.0
        for name, values, color in chart.series:
            share = 100.0 * np.asarray(values, dtype=float) / totals
            ax.bar(x, share, bar_width, bottom=bottom, color=color, edgecolor="white",
                   linewidth=1.2, label=name)
            text_color = "white" if color != COLOR_SIZES[0] else COLOR_TEXT
            for xi, (b, s) in enumerate(zip(bottom, share)):
                if s >= 8:
                    ax.text(xi, b + s / 2, f"{s:.0f}%", ha="center", va="center",
                            fontsize=7, color=text_color)
            bottom += share
        ax.set_ylim(0, 100)
    else:
        _, values, color = chart.series[0]
        colors = chart.bar_colors or [color] * n
        bars = ax.bar(x, values, bar_width, color=colors, edgecolor="white", linewidth=1.2)
        if chart.value_labels:
            for bar, label in zip(bars, chart.value_labels):
                ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=7, color=COLOR_TEXT_2)
        ax.set_ylim(0, max(values, default=1) * 1.12 or 1)
        # Dữ liệu là số đếm -> chỉ đặt mốc nguyên (tránh nhãn trùng "0, 0, 1, 1" khi số nhỏ).
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    ax.set_xticks(x)
    ax.set_xticklabels(chart.categories, rotation=40 if long_labels else 0,
                       ha="right" if long_labels else "center", fontsize=8, color=COLOR_TEXT)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xlabel(chart.xlabel, fontsize=9, color=COLOR_TEXT_2)
    ax.set_ylabel(chart.ylabel, fontsize=9, color=COLOR_TEXT_2)
    ax.tick_params(axis="y", labelsize=8, colors=COLOR_TEXT_2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b5b3ab")

    if chart.stacked_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # Tiêu đề + phụ đề căn trái theo mép hình; chú thích nằm ngay trên vùng vẽ, bên phải.
    fig.text(0.01, 0.975, chart.title, ha="left", va="top", fontsize=13, fontweight="bold",
             color=COLOR_TEXT)
    fig.text(0.01, 0.915, chart.subtitle, ha="left", va="top", fontsize=9, color=COLOR_TEXT_2)
    entries = chart.legend or ([(name, color) for name, _, color in chart.series]
                               if len(chart.series) > 1 else [])
    if entries:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in entries]
        ax.legend(handles, [label for label, _ in entries], loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), ncol=len(entries), frameon=False, fontsize=8,
                  handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


# ============================================================================
# Vẽ biểu đồ - renderer OpenCV/NumPy thuần (dự phòng khi không có matplotlib)
# ============================================================================

_CV_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _bgr(hex_color: str) -> tuple[int, int, int]:
    """Chuyển màu hex '#rrggbb' sang tuple BGR cho OpenCV."""
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def _text_size(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    """Kích thước (rộng, cao) của chuỗi khi vẽ bằng font Hershey."""
    (width, height), baseline = cv2.getTextSize(text, _CV_FONT, scale, thickness)
    return width, height + baseline


def _put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color: str,
              thickness: int = 1, align: str = "left") -> None:
    """Vẽ chữ khử răng cưa; ``align`` = left / center / right theo toạ độ x của ``org``."""
    width, _ = _text_size(text, scale, thickness)
    x = org[0] - (width // 2 if align == "center" else width if align == "right" else 0)
    cv2.putText(img, text, (int(x), int(org[1])), _CV_FONT, scale, _bgr(color), thickness,
                cv2.LINE_AA)


def _put_rotated_text(img: np.ndarray, text: str, anchor: tuple[int, int], scale: float,
                      color: str, angle: float = 40.0) -> None:
    """
    Vẽ chữ xoay ``angle`` độ (ngược chiều kim đồng hồ), đầu PHẢI của chữ neo tại ``anchor``
    (giống ``rotation=40, ha='right'`` của matplotlib). Chữ được vẽ lên patch trắng, xoay bằng
    warpAffine rồi trộn vào canvas bằng phép min (chữ tối trên nền trắng).
    """
    tw, th = _text_size(text, scale)
    pad = 4
    patch = np.full((th + 2 * pad, tw + 2 * pad, 3), 255, np.uint8)
    _put_text(patch, text, (pad, pad + th - 4), scale, color)
    ph, pw = patch.shape[:2]
    rad = math.radians(angle)
    new_w = int(pw * math.cos(rad) + ph * math.sin(rad)) + 2
    new_h = int(pw * math.sin(rad) + ph * math.cos(rad)) + 2
    matrix = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    matrix[0, 2] += new_w / 2 - pw / 2
    matrix[1, 2] += new_h / 2 - ph / 2
    rotated = cv2.warpAffine(patch, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                             borderValue=(255, 255, 255))
    x1, y1 = anchor[0] - new_w + pad, anchor[1]
    x0, y0 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x1 + new_w), min(img.shape[0], y1 + new_h)
    if x2 <= x0 or y2 <= y0:
        return
    region = img[y0:y2, x0:x2]
    np.minimum(region, rotated[y0 - y1:y2 - y1, x0 - x1:x2 - x1], out=region)


def _nice_ticks(max_value: float, target: int = 5, integer: bool = False) -> list[float]:
    """
    Sinh các mốc trục Y "đẹp" (bước 1/2/2.5/5 x 10^k) từ 0 đến >= max_value.
    ``integer=True`` (dữ liệu đếm): bước tối thiểu là 1 và không dùng 2.5 -> mọi mốc là số nguyên.
    """
    if max_value <= 0:
        return [0.0, 1.0]
    raw_step = max_value / target
    magnitude = 10 ** math.floor(math.log10(raw_step))
    multipliers = (1, 2, 5, 10) if integer else (1, 2, 2.5, 5, 10)
    step = next(m * magnitude for m in multipliers if m * magnitude >= raw_step)
    if integer:
        step = max(1, round(step))
    count = math.ceil(max_value / step)
    return [i * step for i in range(count + 1)]


def render_opencv(chart: BarChart, out_path: Path) -> None:
    """Vẽ ``BarChart`` chỉ bằng OpenCV + NumPy: tiêu đề, trục, lưới, cột, nhãn, chú thích."""
    n = len(chart.categories)
    label_scale, small_scale = 0.5, 0.45
    rotate = max((len(c) for c in chart.categories), default=0) > 6
    longest = max((_text_size(c, label_scale)[0] for c in chart.categories), default=0)
    bottom_space = (int(longest * math.sin(math.radians(40))) + 40 if rotate else 40) + 50
    left, right, top = 110, 40, 140
    plot_h = 560
    width = max(1500, 95 * n + left + right)
    height = top + plot_h + bottom_space
    if chart.stacked_percent:
        height += 40   # chỗ cho chú thích phía dưới
    img = np.full((height, width, 3), 255, np.uint8)
    plot_w = width - left - right
    y_axis = top + plot_h

    _put_text(img, chart.title, (24, 46), 0.95, COLOR_TEXT, 2)
    _put_text(img, chart.subtitle, (24, 84), 0.55, COLOR_TEXT_2)

    # Trục Y + lưới ngang.
    if chart.stacked_percent:
        ticks, y_max = [0, 20, 40, 60, 80, 100], 100.0
    else:
        # Trục Y vừa khít dữ liệu (+12% chỗ cho nhãn giá trị), chỉ vẽ các mốc "đẹp" bên dưới.
        y_max = max(max(chart.series[0][1], default=0), 1) * 1.12
        ticks = [t for t in _nice_ticks(y_max, integer=True) if t <= y_max]
    for tick in ticks:
        y = int(y_axis - plot_h * tick / y_max)
        cv2.line(img, (left, y), (width - right, y), _bgr(COLOR_GRID), 1, cv2.LINE_AA)
        label = f"{tick:.0f}%" if chart.stacked_percent else f"{tick:,.0f}"
        _put_text(img, label, (left - 10, y + 5), small_scale, COLOR_TEXT_2, align="right")
    cv2.line(img, (left, y_axis), (width - right, y_axis), _bgr("#b5b3ab"), 1, cv2.LINE_AA)

    # Cột (có khe trắng 2px giữa các đoạn chồng).
    slot = plot_w / max(n, 1)
    bar_w = max(4, int(slot * 0.72))
    totals = (np.sum([vals for _, vals, _ in chart.series], axis=0).astype(float)
              if chart.stacked_percent else None)
    for i, category in enumerate(chart.categories):
        cx = int(left + slot * (i + 0.5))
        x0, x1 = cx - bar_w // 2, cx + bar_w // 2
        if chart.stacked_percent:
            base = 0.0
            total = totals[i] or 1.0
            for s_index, (_, vals, color) in enumerate(chart.series):
                share = 100.0 * vals[i] / total
                y_top = int(y_axis - plot_h * (base + share) / 100)
                y_bot = int(y_axis - plot_h * base / 100)
                if y_bot - y_top > 2:
                    cv2.rectangle(img, (x0, y_top + 1), (x1, y_bot - 1), _bgr(color), -1)
                if share >= 8:
                    text_color = COLOR_TEXT if s_index == 0 else "#ffffff"
                    _put_text(img, f"{share:.0f}%", (cx, (y_top + y_bot) // 2 + 5), 0.42,
                              text_color, align="center")
                base += share
        else:
            value = chart.series[0][1][i]
            color = (chart.bar_colors or [chart.series[0][2]] * n)[i]
            y_top = int(y_axis - plot_h * value / y_max)
            cv2.rectangle(img, (x0, y_top), (x1, y_axis - 1), _bgr(color), -1)
            if chart.value_labels:
                _put_text(img, chart.value_labels[i], (cx, y_top - 8), 0.42, COLOR_TEXT_2,
                          align="center")
        if rotate:
            _put_rotated_text(img, category, (cx + 6, y_axis + 8), label_scale, COLOR_TEXT)
        else:
            _put_text(img, category, (cx, y_axis + 26), label_scale, COLOR_TEXT, align="center")

    # Nhãn trục.
    _put_text(img, chart.xlabel, (left + plot_w // 2, height - 18 - (40 if chart.stacked_percent
                                                                   else 0)),
              0.55, COLOR_TEXT_2, align="center")
    ylabel_patch = np.full((30, _text_size(chart.ylabel, 0.55)[0] + 10, 3), 255, np.uint8)
    _put_text(ylabel_patch, chart.ylabel, (5, 21), 0.55, COLOR_TEXT_2)
    ylabel_patch = cv2.rotate(ylabel_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    y0 = max(top, top + plot_h // 2 - ylabel_patch.shape[0] // 2)
    y1 = min(y_axis, y0 + ylabel_patch.shape[0])
    np.minimum(img[y0:y1, 14:14 + ylabel_patch.shape[1]], ylabel_patch[: y1 - y0],
               out=img[y0:y1, 14:14 + ylabel_patch.shape[1]])

    # Chú thích: hàng trên-phải (cột đơn) hoặc hàng dưới (cột chồng).
    entries = chart.legend or [(name, color) for name, _, color in chart.series]
    if len(entries) > 1 or chart.legend:
        widths = [_text_size(label, small_scale)[0] + 44 for label, _ in entries]
        x = width - right - sum(widths) if not chart.stacked_percent else (
            (width - sum(widths)) // 2)
        y = 118 if not chart.stacked_percent else height - 24
        for (label, color), w in zip(entries, widths):
            cv2.rectangle(img, (x, y - 12), (x + 14, y + 2), _bgr(color), -1)
            _put_text(img, label, (x + 22, y), small_scale, COLOR_TEXT)
            x += w
    ok, encoded = cv2.imencode(".png", img)
    if ok:
        encoded.tofile(str(out_path))


# ============================================================================
# Xây dựng các biểu đồ từ summary
# ============================================================================

def _class_subtitle(total_boxes: int, total_images: int, target: dict[str, Any]) -> str:
    """
    Phụ đề biểu đồ class: tổng box gốc và cách chúng được chia sau khi lọc. Các phần cộng lại
    đúng bằng tổng (kept + dropped + box lỗi của class 0-9 + box trên ảnh không dùng được).
    """
    parts = [f"{fmt_int(target['kept_boxes_total'])} kept (0-9)",
             f"{fmt_int(target['dropped_boxes_total'])} dropped (>=10)"]
    if target["rejected_kept_class_boxes"]:
        parts.append(f"{fmt_int(target['rejected_kept_class_boxes'])} invalid boxes of 0-9")
    if target["boxes_on_unusable_images"]:
        parts.append(f"{fmt_int(target['boxes_on_unusable_images'])} on unusable images")
    return (f"{fmt_int(total_boxes)} boxes in {fmt_int(total_images)} images -> after "
            f"filtering: " + ", ".join(parts))


def build_charts(summary: dict[str, Any]) -> list[BarChart]:
    """Tạo mô tả 5 biểu đồ (độc lập renderer) từ dict thống kê."""
    classes = summary["classes_raw"]["classes"]
    overview = summary["overview"]
    target = summary["target_view"]
    charts: list[BarChart] = []

    # 1) Phân bố class: class giữ lại (0..9) / bị loại (>=10) / không khai báo khác màu.
    def class_color(row: dict[str, Any]) -> str:
        if not row["declared"]:
            return COLOR_UNDECLARED
        return COLOR_KEPT if row["status"] == "kept" else COLOR_DROPPED

    legend = [("Kept (0-9)", COLOR_KEPT), ("Dropped (>=10)", COLOR_DROPPED)]
    if any(not r["declared"] for r in classes):
        legend.append(("Undeclared in yaml", COLOR_UNDECLARED))
    charts.append(BarChart(
        filename="class_distribution.png",
        title="Class distribution (raw labels)",
        subtitle=_class_subtitle(summary["classes_raw"]["total_boxes"],
                                 overview["total_images"], target),
        xlabel="Class (original id, never re-indexed)", ylabel="Number of bounding boxes",
        categories=[f"{r['id']}: {r['name']}" for r in classes],
        series=[("boxes", [r["boxes"] for r in classes], COLOR_KEPT)],
        bar_colors=[class_color(r) for r in classes], legend=legend,
        value_labels=[fmt_int(r["boxes"]) for r in classes],
    ))

    # 2) Độ phân giải: top-N + "Other".
    res = summary["resolution"]
    res_rows = [(r["resolution"], r["count"], r["percent"]) for r in res["top"]]
    if res["other"]["count"]:
        res_rows.append((f"Other ({res['other']['resolutions']})", res["other"]["count"],
                         res["other"]["percent"]))
    charts.append(BarChart(
        filename="resolution_distribution.png",
        title="Image resolution distribution (W x H)",
        subtitle=(f"{fmt_int(res['images_with_known_size'])} images, "
                  f"{res['unique_resolutions']} distinct resolutions"),
        xlabel="Resolution (width x height, px)", ylabel="Number of images",
        categories=[r[0] for r in res_rows],
        series=[("images", [r[1] for r in res_rows], COLOR_SINGLE)],
        value_labels=[f"{fmt_int(r[1])} ({r[2]:.1f}%)" for r in res_rows],
    ))

    # 3) Mật độ box mỗi ảnh.
    density = summary["density"]
    charts.append(BarChart(
        filename="boxes_per_image.png",
        title="Bounding boxes per image (all classes)",
        subtitle=(f"mean {density['mean_all_images']:.2f} over all images, "
                  f"{density['mean_labeled_images']:.2f} over labeled images, "
                  f"max {density['max']}"),
        xlabel="Boxes in image", ylabel="Number of images",
        categories=list(density["histogram"]),
        series=[("images", list(density["histogram"].values()), COLOR_SINGLE)],
        value_labels=[fmt_int(v) for v in density["histogram"].values()],
    ))

    # 4) Kích thước box theo class (cột chồng 100%), cột đầu tiên = toàn bộ.
    size_overall = summary["classes_raw"]["size_overall"]
    size_rows = [("All", size_overall)] + [
        (f"{r['id']}: {r['name']}", r["size"]) for r in classes if sum(r["size"].values())]
    charts.append(BarChart(
        filename="bbox_size_distribution.png",
        title="Bounding-box size by class (COCO area buckets)",
        subtitle=("small < 32x32 px, medium < 96x96 px, large otherwise - overall: "
                  + ", ".join(f"{b} {summary['classes_raw']['size_overall_percent'][b]:.1f}%"
                              for b in SIZE_BUCKETS)),
        xlabel="Class", ylabel="Share of valid boxes",
        categories=[name for name, _ in size_rows],
        series=[(bucket, [row[bucket] for _, row in size_rows], color)
                for bucket, color in zip(SIZE_BUCKETS, COLOR_SIZES)],
        stacked_percent=True,
    ))

    # 5) Số bản sao Roboflow trên mỗi frame gốc.
    leak = summary["leakage"]
    hist = leak["copies_per_stem_histogram"]
    charts.append(BarChart(
        filename="copies_per_stem.png",
        title="Copies per source frame (Roboflow augmentation)",
        subtitle=(f"{fmt_int(leak['unique_source_stems'])} source frames -> "
                  f"{fmt_int(leak['total_files'])} files; "
                  f"{fmt_int(leak['stems_in_multiple_splits'])} frames appear in >1 original "
                  "split"),
        xlabel="Files sharing the same source frame", ylabel="Number of source frames",
        categories=list(hist),
        series=[("frames", list(hist.values()), COLOR_SINGLE)],
        value_labels=[fmt_int(v) for v in hist.values()],
    ))
    return charts


def save_charts(summary: dict[str, Any], out_dir: Path, use_matplotlib: bool) -> dict[str, Any]:
    """Vẽ và lưu toàn bộ biểu đồ; trả về renderer đã dùng + danh sách file."""
    plt = _load_pyplot() if use_matplotlib else None
    files = []
    for chart in build_charts(summary):
        out_path = out_dir / chart.filename
        if plt is not None:
            render_matplotlib(chart, out_path, plt)
        else:
            render_opencv(chart, out_path)
        files.append(out_path.name)
    return {"renderer": "matplotlib" if plt is not None else "opencv", "files": files}


# ============================================================================
# Ghi file kết quả & in báo cáo console
# ============================================================================

def write_class_csv(summary: dict[str, Any], out_path: Path) -> None:
    """Ghi bảng class (raw view) ra CSV - tiện chuyển thành bảng LaTeX."""
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "name", "declared", "status", "boxes", "images", "percent_boxes",
                         "valid_boxes", "small", "medium", "large", "median_side_px"])
        for row in summary["classes_raw"]["classes"]:
            writer.writerow([row["id"], row["name"], row["declared"], row["status"], row["boxes"],
                             row["images"], row["percent_boxes"], row["valid_boxes"],
                             *(row["size"][b] for b in SIZE_BUCKETS), row["median_side_px"]])


def format_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Bảng chữ căn cột đơn giản; cột số căn phải."""
    rows = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    numeric = [all(re.fullmatch(r"[-+]?[\d,.]+%?|None|-", r[i]) for r in rows) if rows else False
               for i in range(len(headers))]

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(c.rjust(w) if num else c.ljust(w)
                         for c, w, num in zip(cells, widths, numeric))

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines += [fmt(r) for r in rows]
    return "\n".join("  " + line for line in lines)


def print_report(summary: dict[str, Any]) -> None:
    """In báo cáo EDA dễ đọc ra console (có thể copy vào báo cáo)."""
    ov = summary["overview"]
    bar = "=" * 78
    print(f"\n{bar}\nBÁO CÁO EDA - {summary['meta']['raw_dir']}\n{bar}")
    print(f"Định dạng nhãn: {summary['format_detection']['detected'].upper()} "
          f"{summary['format_detection']['file_counts']}")
    print(f"File tên class: {summary['meta']['class_names_source']}")

    print("\n[1] Tổng quan")
    print(f"  Tổng số ảnh           : {fmt_int(ov['total_images'])} ({ov['total_size_mb']} MB)")
    print(f"  Có file nhãn / không  : {fmt_int(ov['images_with_label_file'])} / "
          f"{fmt_int(ov['images_without_label_file'])}")
    print(f"  File nhãn rỗng        : {ov['empty_label_files']}")
    print(f"  File nhãn mồ côi      : {ov['orphan_label_files']['count']}")
    print(f"  Ảnh 0 byte            : {ov['zero_byte_images']['count']}")
    corrupted = ov["corrupted_images"]
    note = "" if corrupted["fully_verified"] else " (cận dưới - chạy với --no-verify-decode)"
    print(f"  Ảnh hỏng (cv2 lỗi)    : {corrupted['count']}{note}"
          + (f"  {corrupted['examples']}" if corrupted["count"] else ""))
    unreadable = ov["unreadable_files"]
    if unreadable["count"]:
        print(f"  (!) File không đọc được (lỗi IO, nên chạy lại): {unreadable['count']}  "
              f"{unreadable['examples'][:2]}")
    print(f"  JPEG không kết thúc bằng EOI (cắt cụt / dữ liệu thừa): {ov['jpeg_missing_eoi']}"
          f"   |  Ảnh xoay EXIF: {ov['exif_rotated_images']}")
    print(format_table(["split", "images", "with_label", "no_label", "empty_label", "boxes"],
                       [[s, fmt_int(v.get("images", 0)), fmt_int(v.get("with_label_file", 0)),
                         fmt_int(v.get("without_label_file", 0)),
                         fmt_int(v.get("empty_label_files", 0)), fmt_int(v.get("boxes", 0))]
                        for s, v in ov["images_per_split"].items()]))

    res = summary["resolution"]
    print(f"\n[2] Độ phân giải ({res['unique_resolutions']} loại, hướng: {res['orientation']})")
    rows = [[r["resolution"], fmt_int(r["count"]), f"{r['percent']:.2f}%"] for r in res["top"]]
    if res["other"]["count"]:
        rows.append([f"khác ({res['other']['resolutions']} loại)",
                     fmt_int(res["other"]["count"]), f"{res['other']['percent']:.2f}%"])
    print(format_table(["W x H", "images", "share"], rows))

    cls = summary["classes_raw"]
    print(f"\n[3] Bounding box theo class - toàn bộ nhãn gốc ({fmt_int(cls['total_boxes'])} box)")
    print(format_table(
        ["id", "name", "status", "boxes", "images", "%boxes", "small", "medium", "large",
         "med_side"],
        [[r["id"], r["name"] + ("" if r["declared"] else " (!)"), r["status"],
          fmt_int(r["boxes"]), fmt_int(r["images"]), f"{r['percent_boxes']:.2f}%",
          *(fmt_int(r["size"][b]) for b in SIZE_BUCKETS), r["median_side_px"]]
         for r in cls["classes"]]))
    if cls["undeclared_classes"]:
        print(f"  (!) Class có trong nhãn nhưng KHÔNG khai báo trong yaml: "
              f"{cls['undeclared_classes']}")
    print(f"  Kích thước box tổng thể: {cls['size_overall_percent']} "
          f"(cạnh trung vị {cls['median_side_px']} px)")

    den = summary["density"]
    print(f"\n[4] Mật độ box/ảnh: mean {den['mean_all_images']} (ảnh có nhãn: "
          f"{den['mean_labeled_images']}), median {den['median_all_images']}, max {den['max']}")
    print("  histogram: " + ", ".join(f"{k}:{v}" for k, v in den["histogram"].items()))

    quality = summary["label_quality"]
    print("\n[5] Chất lượng nhãn")
    if quality["issues"]:
        for kind, entry in quality["issues"].items():
            print(f"  {kind:<20}: {fmt_int(entry['count'])}  ví dụ: {entry['examples'][:2]}")
    else:
        print("  Không phát hiện dòng nhãn lỗi.")

    leak = summary["leakage"]
    print("\n[6] Leakage / chuỗi frame")
    print(f"  Frame gốc (stem) duy nhất  : {fmt_int(leak['unique_source_stems'])} cho "
          f"{fmt_int(leak['total_files'])} file (TB {leak['copies_per_stem_mean']} bản/stem, "
          f"tối đa {leak['copies_per_stem_max']})")
    print(f"  Stem nằm ở >1 split gốc    : {fmt_int(leak['stems_in_multiple_splits'])} "
          f"({fmt_int(leak['files_of_leaking_stems'])} file) {leak['leaking_split_combinations']}")
    print(f"  Chuỗi video ở >1 split gốc : {leak['sequences_in_multiple_splits']}/"
          f"{leak['sequences']} chuỗi ({fmt_int(leak['files_in_multi_split_sequences'])} file)"
          " - ước lượng THÔ: họ tên chung (frame_#_jpg...) gộp nhiều video vào một chuỗi")
    dup = leak["exact_duplicates"]
    if dup is None:
        print("  Ảnh trùng nội dung (md5)   : không kiểm tra (--no-verify-decode)")
    else:
        print(f"  Ảnh trùng nội dung (md5)   : {dup['groups']} nhóm ({dup['files_in_groups']} "
              f"file), {dup['groups_across_splits']} nhóm ở >1 split, trong đó "
              f"{dup['groups_across_splits_with_different_stems']} nhóm khác tên gốc")
    print(format_table(["family", "stems", "files"],
                       [[f["family"], fmt_int(f["stems"]), fmt_int(f["files"])]
                        for f in leak["top_families"]]))

    tv = summary["target_view"]
    print("\n[7] Target view - CHỈ GIỮ class 0..9 (giữ nguyên ID), bỏ class >= 10")
    print(format_table(["id", "name", "boxes", "images", "%boxes"],
                       [[r["id"], r["name"], fmt_int(r["boxes"]), fmt_int(r["images"]),
                         f"{r['percent_boxes']:.2f}%"] for r in tv["kept_classes"]]))
    print(f"  Box giữ lại / bị loại     : {fmt_int(tv['kept_boxes_total'])} / "
          f"{fmt_int(tv['dropped_boxes_total'])}  (mất cân bằng max/min = "
          f"{tv['imbalance_ratio_max_min']})")
    print(f"  Box class 0..9 bị lỗi     : {fmt_int(tv['rejected_kept_class_boxes'])}   |  "
          f"box trên ảnh không dùng được: {fmt_int(tv['boxes_on_unusable_images'])}")
    print(f"  Ảnh không dùng được (bỏ)  : {fmt_int(tv['unusable_images_excluded'])} "
          f"(0 byte / không giải mã được / không đọc được)")
    print(f"  Ảnh còn object (positive) : {fmt_int(tv['positive_images'])} "
          f"{tv['positive_images_per_split']}")
    print(f"  Ảnh thành background      : {fmt_int(tv['background_images_after_filter'])} "
          f"{tv['background_reasons']}")
    print(f"  Ảnh mơ hồ (bỏ, không làm negative): {fmt_int(tv['ambiguous_images_excluded'])}")
    print(f"  -> giữ 10% làm background negatives ≈ {fmt_int(tv['background_kept_if_10_percent'])}"
          f" ảnh (ước lượng, quy tắc box xấp xỉ clean_and_filter.py)")
    print(f"\nĐã lưu kết quả vào: {summary['meta']['out_dir']}")
    print(f"Biểu đồ ({summary['plots']['renderer']}): {', '.join(summary['plots']['files'])}")
    print(f"Thời gian chạy: {summary['meta']['runtime_seconds']} s\n{bar}")


# ============================================================================
# Hàm chính (public API)
# ============================================================================

def run_eda(raw_dir: Path = DEFAULT_RAW_DIR, out_dir: Path = DEFAULT_OUT_DIR,
            verify_decode: bool = True, use_matplotlib: bool = True,
            workers: int | None = None, top_n: int = 10) -> dict[str, Any]:
    """
    Chạy toàn bộ EDA trên ``raw_dir`` và ghi kết quả vào ``out_dir``.

    Trả về dict tổng hợp (cũng được ghi thành ``eda_summary.json``). Nếu định dạng nhãn không
    phải YOLO thì chỉ báo cáo định dạng và dừng (không lỗi).
    """
    configure_console()
    started = time.perf_counter()
    raw_dir, out_dir = Path(raw_dir).resolve(), Path(out_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu thô: {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or min(32, (os.cpu_count() or 4) * 2)

    print(f"[eda] Quét thư mục {raw_dir} ...", flush=True)
    _clear_previous_outputs(out_dir)
    # --out nằm trong raw_dir -> bỏ qua khi quét để biểu đồ cũ không bị đếm thành ảnh.
    scan = scan_dataset(raw_dir, skip_dir=out_dir if out_dir.is_relative_to(raw_dir) else None)
    names, declared, yaml_path = load_class_names(raw_dir, scan.yaml_files,
                                                  scan.classes_txt_files)
    meta = {
        "raw_dir": raw_dir.as_posix(),
        "out_dir": out_dir.as_posix(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "class_names_source": yaml_path.as_posix() if yaml_path else "built-in default",
        "kept_class_ids": sorted(KEPT_CLASS_IDS),
        "verify_decode": verify_decode,
        "workers": workers,
    }
    summary: dict[str, Any] = {"meta": meta, "format_detection": scan.format_info}

    detected = scan.format_info["detected"]
    # Chỉ phân tích chi tiết YOLO. Bộ ảnh chưa có file nhãn nào ("none") vẫn được phân tích
    # như YOLO 0 nhãn; dừng sớm khi là COCO / VOC hoặc không có ảnh.
    if detected in ("coco", "voc") or not scan.images:
        print(f"[eda] Định dạng nhãn phát hiện: {detected.upper()} "
              f"{scan.format_info['file_counts']}, {len(scan.images)} ảnh.")
        print("[eda] Script chỉ phân tích chi tiết định dạng YOLO (.txt) - dừng tại đây.")
        meta["runtime_seconds"] = round(time.perf_counter() - started, 2)
        _write_json(summary, out_dir / "eda_summary.json")
        return summary
    if detected == "none":
        print("[eda] Không có file nhãn nào - phân tích như bộ YOLO chưa gán nhãn (mọi ảnh là "
              "background).")

    print(f"[eda] {len(scan.images):,} ảnh, {len(declared)} class khai báo. "
          f"Đang kiểm tra ảnh + nhãn ({workers} thread, verify_decode={verify_decode}) ...",
          flush=True)
    records = inspect_all(scan, verify_decode, workers)

    summary.update({
        "class_names": {str(k): v for k, v in sorted(names.items())},
        "overview": compute_overview(records, scan, raw_dir, verify_decode),
        "resolution": compute_resolution_stats(records, top_n),
        "classes_raw": compute_class_stats(records, names, declared),
        "density": compute_density_stats(records),
        "label_quality": {"issues": _issue_examples(records, raw_dir)},
        "leakage": compute_leakage_stats(records, top_n),
        "target_view": compute_target_view(records, names),
    })
    summary["plots"] = save_charts(summary, out_dir, use_matplotlib)
    write_class_csv(summary, out_dir / "class_distribution.csv")
    meta["runtime_seconds"] = round(time.perf_counter() - started, 2)
    _write_json(summary, out_dir / "eda_summary.json")
    print_report(summary)
    return summary


def _write_json(data: dict[str, Any], path: Path) -> None:
    """Ghi JSON UTF-8, thụt lề 2, giữ nguyên tiếng Việt."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _clear_previous_outputs(out_dir: Path) -> None:
    """Xoá các file kết quả của lần chạy trước (chỉ những tên do script này ghi ra) để thư mục
    --out không lẫn biểu đồ/CSV cũ không khớp với eda_summary.json mới."""
    for name in OUTPUT_FILES:
        (out_dir / name).unlink(missing_ok=True)


def _positive_int(text: str) -> int:
    """Kiểu argparse: số nguyên dương (cho --workers, --top-n)."""
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value <= 0:
        raise argparse.ArgumentTypeError(f"phải là số nguyên dương, nhận được '{text}'")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Định nghĩa tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="EDA cho bộ dữ liệu YOLO thô (thống kê class, độ phân giải, lỗi nhãn, "
                    "leakage) + biểu đồ PNG.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                        help=f"thư mục dữ liệu thô, quét đệ quy (mặc định: {DEFAULT_RAW_DIR})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"thư mục lưu báo cáo/biểu đồ (mặc định: {DEFAULT_OUT_DIR})")
    parser.add_argument("--top-n", type=_positive_int, default=10,
                        help="số dòng top cho bảng độ phân giải / họ tên file (mặc định: 10)")
    parser.add_argument("--workers", type=_positive_int, default=None,
                        help="số thread đọc ảnh (mặc định: 2 x số CPU, tối đa 32)")
    parser.add_argument("--verify-decode", action=argparse.BooleanOptionalAction, default=True,
                        help="giải mã toàn bộ mỗi ảnh để phát hiện ảnh hỏng + tính md5 (mặc định: "
                             "bật). --no-verify-decode chỉ đọc header: nhanh hơn nhưng số ảnh "
                             "hỏng chỉ là cận dưới")
    parser.add_argument("--matplotlib", dest="use_matplotlib",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="vẽ biểu đồ bằng matplotlib nếu có (mặc định: bật). "
                             "--no-matplotlib vẽ bằng OpenCV thuần")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào dòng lệnh. Mã thoát: 0 thành công, 2 lỗi đường dẫn / ghi file."""
    configure_console()   # trước parse_args: --help tiếng Việt không lỗi trên console cp1252
    args = parse_args(argv)
    if not args.raw_dir.is_dir():
        print(f"[eda] LỖI: Không tìm thấy thư mục dữ liệu thô: {args.raw_dir}", file=sys.stderr)
        return 2
    try:
        run_eda(raw_dir=args.raw_dir, out_dir=args.out, verify_decode=args.verify_decode,
                use_matplotlib=args.use_matplotlib, workers=args.workers, top_n=args.top_n)
    except OSError as error:   # ví dụ --out trỏ vào một file đã tồn tại, ổ đĩa đầy...
        print(f"[eda] LỖI: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
