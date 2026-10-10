"""Tạo tập dữ liệu hiệu chuẩn (Calibration Dataset) cho PTQ INT8 trên Hailo-8.

Post-Training Quantization cần một tập ảnh "tiêu biểu" (không cần nhãn) để Hailo
Dataflow Compiler (DFC) chạy thử mô hình FP32, ghi lại dải giá trị (min/max,
histogram) của activation ở từng layer, rồi chọn scale/zero-point INT8 phù hợp.
Ảnh hiệu chuẩn phải được tiền xử lý GIỐNG HỆT lúc suy luận trên thiết bị biên,
nếu không dải activation đo được sẽ lệch và mAP sau lượng tử hóa sụt mạnh.

Luồng chạy (hàm ``main``):
    1. Liệt kê ảnh trong ``--data-dir`` (mặc định data/processed/Train/images),
       bỏ qua ảnh augment offline ``*_aug###`` (xem ``list_images``).
    2. Chọn ngẫu nhiên có seed (42) đúng ``--calib-size`` ảnh; ảnh hỏng được
       thay bằng ảnh kế tiếp trong thứ tự xáo trộn -> luôn đủ số lượng.
    3. Tiền xử lý như Ultralytics/YOLO26: Letterbox về 640x640 giữ tỉ lệ khung
       hình (pad 114), BGR -> RGB, layout NHWC.
    4. Ghi ``models/calibration_data.npy`` (N, 640, 640, 3) uint8 [0, 255]
       (mặc định) hoặc float32 [0, 1] với ``--normalize``, kèm file manifest JSON.

Ví dụ:
    python optimization/calibrate_ptq.py
    python optimization/calibrate_ptq.py --calib-size 500 --output models/calib_500.npy
    python optimization/calibrate_ptq.py --normalize   # float32 [0,1] (không khuyến nghị)

Sử dụng với Hailo DFC (Giai đoạn 3, compile_npu.py):
    runner.optimize(np.load("models/calibration_data.npy"))
    # hoặc CLI: hailo optimize yolo26m.har --calib-set-path models/calibration_data.npy
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

import cv2
import numpy as np

class _SimpleProgress:
    """Thay thế tối giản cho tqdm (in tiến trình mỗi 10%), dùng khi chưa cài tqdm.

    Hỗ trợ đúng hai kiểu dùng trong file: ``with tqdm(total=n) as bar: bar.update(1)``
    và duyệt ``for x in tqdm(iterable)``.
    """

    def __init__(self, iterable: Iterable | None = None, total: int | None = None,
                 desc: str = "", **_: object) -> None:
        self.iterable = iterable
        self.total = total if total is not None else len(iterable)  # type: ignore[arg-type]
        self.desc, self.n = desc, 0
        self._step = max(1, self.total // 10)

    def __iter__(self) -> Iterator:
        for item in self.iterable or ():
            yield item
            self.update(1)

    def update(self, k: int = 1) -> None:
        self.n += k
        if self.n % self._step == 0 or self.n == self.total:
            print(f"{self.desc}: {self.n}/{self.total}", flush=True)

    def __enter__(self) -> "_SimpleProgress":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


try:
    from tqdm import tqdm
except ImportError:  # tqdm không bắt buộc: Ultralytics >= 8.4 không còn kéo theo tqdm
    tqdm = _SimpleProgress

T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "Train" / "images"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "calibration_data.npy"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAD_VALUE = 114  # màu xám pad mặc định của Ultralytics (LetterBox.padding_value)
# Ảnh do data_engineering/augmentations.py sinh thêm: "<stem>_aug001.jpg"
AUG_PATTERN = re.compile(r"_aug\d{3}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo tập calibration cho PTQ INT8 (Hailo DFC).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Thư mục ảnh nguồn (chỉ đọc ảnh, không đọc nhãn)")
    parser.add_argument("--calib-size", type=int, default=300,
                        help="Số ảnh hiệu chuẩn (đề bài: 100-500)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="File .npy đầu ra")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước đầu vào vuông của mô hình (bội số của 32)")
    parser.add_argument("--seed", type=int, default=42, help="Seed chọn ảnh")
    parser.add_argument("--normalize", action="store_true",
                        help="Xuất float32 [0,1] thay vì uint8 [0,255] (mặc định uint8)")
    parser.add_argument("--include-aug", action="store_true",
                        help="Cho phép chọn cả ảnh augment offline *_aug###")
    parser.add_argument("--workers", type=int, default=8,
                        help="Số luồng đọc/tiền xử lý ảnh")
    return parser.parse_args()


# ===========================================================================
# 1. Tiền xử lý (dùng chung cho calibration, kiểm thử ONNX và ứng dụng Edge)
# ===========================================================================
def letterbox(image: np.ndarray, new_size: int = 640, pad_value: int = PAD_VALUE,
              scaleup: bool = True) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize giữ tỉ lệ khung hình rồi pad thành ảnh vuông ``new_size`` x ``new_size``.

    Sao chép đúng ``ultralytics.data.augment.LetterBox`` (auto=False, center=True,
    INTER_LINEAR) - chế độ dùng khi suy luận với đầu vào cố định như NPU:
      - r = min(new/h, new/w): cạnh dài nhất vừa khít khung, không méo hình
        (biển báo tròn vẫn tròn -> đặc trưng hình học giống lúc huấn luyện).
      - Phần thiếu được pad đều hai phía bằng màu xám 114; quy tắc làm tròn
        round(d - 0.1) / round(d + 0.1) giống Ultralytics để lệch pad 1 px
        (nếu có) dồn về phía dưới/phải.
    Trả về (ảnh, tỉ lệ r, (pad_left, pad_top)) - cần cho bước khôi phục tọa độ
    box về ảnh gốc ở Giai đoạn 4: x_goc = (x - pad_left) / r.
    """
    h, w = image.shape[:2]
    r = min(new_size / h, new_size / w)
    if not scaleup:
        r = min(r, 1.0)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (new_size - new_w) / 2, (new_size - new_h) / 2

    if (w, h) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                               value=(pad_value, pad_value, pad_value))
    return image, r, (left, top)


def read_image(path: Path) -> np.ndarray:
    """Đọc ảnh BGR an toàn với đường dẫn Unicode trên Windows; lỗi -> ValueError."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError("file rỗng (0 byte)")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)  # ép 3 kênh (ảnh xám/RGBA -> BGR)
    if image is None:
        raise ValueError("không giải mã được (ảnh hỏng hoặc sai định dạng)")
    return image


def preprocess(path: Path, imgsz: int) -> np.ndarray:
    """Đọc ảnh -> letterbox -> RGB, trả về mảng HWC uint8 (imgsz, imgsz, 3).

    BGR -> RGB vì Ultralytics huấn luyện với ảnh RGB (OpenCV đọc mặc định BGR).
    Không chia 255 ở đây: phép chuẩn hóa được giao cho NPU (xem ``to_output_dtype``).
    """
    image, _, _ = letterbox(read_image(path), imgsz)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_output_dtype(batch: np.ndarray, normalize: bool) -> np.ndarray:
    """Chuyển batch uint8 sang dtype đầu ra.

    VÌ SAO HAILO DFC KHUYẾN NGHỊ uint8 [0, 255]:
      - Mô hình YOLO chỉ chuẩn hóa đầu vào bằng x/255 (mean=0, std=255). Trong
        model script (.alls) của DFC ta khai báo
            normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
        -> phép chia được GỘP vào layer đầu tiên chạy trên NPU. Khi đó dữ liệu
        hiệu chuẩn phải ở dạng TRƯỚC chuẩn hóa, tức uint8 0..255 giống frame camera.
      - Ảnh camera vốn đã là số nguyên 8-bit -> lượng tử hóa layer đầu vào là
        "không mất mát" (scale = 1, zero-point = 0), không cộng thêm sai số.
      - Khi chạy trên Raspberry Pi 5, CPU gửi thẳng frame uint8 qua PCIe cho
        Hailo-8: ít hơn 4 lần băng thông so với float32 (PCIe của Pi 5 chỉ x1 -
        một nút thắt thật sự) và không tốn thời gian CPU chuyển sang float.
      - float32 [0, 1] (``--normalize``) chỉ dùng khi KHÔNG khai báo normalization
        trong .alls; cách này vẫn chạy được nhưng phải chia 255 trên CPU mỗi frame.
    """
    if normalize:
        return batch.astype(np.float32) / 255.0
    return batch


# ===========================================================================
# 2. Chọn ảnh hiệu chuẩn
# ===========================================================================
def natural_key(path: Path) -> list[object]:
    """Khóa sắp xếp alpha-numeric ("frame_2" < "frame_10"), độc lập hệ điều hành."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_images(data_dir: Path, include_aug: bool) -> list[Path]:
    """Liệt kê ảnh theo thứ tự cố định (để seed cho kết quả giống nhau mọi máy).

    Mặc định bỏ ảnh ``*_aug###`` (sinh bởi augmentations.py để cân bằng lớp):
    chúng có viền pad do affine, motion blur nhân tạo... - không đại diện cho
    phân phối frame camera thật mà NPU sẽ nhận, dễ làm lệch dải activation.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {data_dir}")
    images = [p for p in data_dir.iterdir()
              if p.suffix.lower() in IMAGE_EXTS and (include_aug or not AUG_PATTERN.search(p.stem))]
    return sorted(images, key=natural_key)


def chunks(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def build_calibration_set(candidates: list[Path], n: int, imgsz: int, normalize: bool,
                          output: Path, workers: int) -> tuple[list[Path], list[str]]:
    """Tiền xử lý tới khi đủ ``n`` ảnh hợp lệ, ghi trực tiếp vào file .npy.

    ``candidates`` đã được xáo trộn theo seed. Ta xử lý theo đợt: đợt đầu lấy n
    ảnh đầu tiên; ảnh nào hỏng thì đợt sau lấy thêm từ phần còn lại -> kết quả
    vẫn tất định và luôn đủ ``n`` ảnh (nếu thư mục có đủ ảnh tốt).
    Dùng ``np.lib.format.open_memmap`` để ghi dần ra đĩa: 300 ảnh float32
    640x640x3 chiếm ~1.5 GB, không cần giữ toàn bộ trong RAM.
    """
    dtype = np.float32 if normalize else np.uint8
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(output, mode="w+", dtype=dtype, shape=(n, imgsz, imgsz, 3))
    chosen: list[Path] = []
    skipped: list[str] = []
    cursor = 0

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, \
                tqdm(total=n, desc="Calibration", unit="img") as bar:
            while len(chosen) < n:
                need = n - len(chosen)
                batch_paths = candidates[cursor:cursor + need]
                cursor += need
                if not batch_paths:
                    raise RuntimeError(f"Chỉ có {len(chosen)} ảnh hợp lệ, cần {n}. "
                                       "Giảm --calib-size hoặc kiểm tra thư mục ảnh.")
                # Duyệt kết quả theo đúng thứ tự đầu vào -> tất định dù chạy đa luồng.
                futures = [pool.submit(preprocess, p, imgsz) for p in batch_paths]
                for path, future in zip(batch_paths, futures):
                    try:
                        image = future.result()
                    except Exception as exc:  # ảnh hỏng: ghi lý do, lấy ảnh thay thế
                        skipped.append(f"{path.name}: {exc}")
                        continue
                    array[len(chosen)] = to_output_dtype(image, normalize)
                    chosen.append(path)
                    bar.update(1)
        array.flush()
    finally:
        # Giải phóng memmap kể cả khi lỗi: trên Windows file đang map không xóa được.
        del array
    return chosen, skipped


def channel_stats(output: Path, chunk: int = 32) -> dict[str, list[float]]:
    """Mean/std theo kênh RGB (thang 0..255) của tập calibration - dùng cho báo cáo.

    Cộng dồn theo từng nhóm ảnh để không phải nạp cả mảng (float32 ~1.5 GB) vào RAM.
    """
    data = np.load(output, mmap_mode="r")
    scale = 255.0 if data.dtype == np.float32 else 1.0
    total, total_sq, count = np.zeros(3), np.zeros(3), 0
    for start in range(0, len(data), chunk):
        block = data[start:start + chunk].reshape(-1, 3).astype(np.float64) * scale
        total += block.sum(axis=0)
        total_sq += np.square(block).sum(axis=0)
        count += block.shape[0]
    del data
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 0.0))
    return {"mean_rgb": [round(float(v), 3) for v in mean],
            "std_rgb": [round(float(v), 3) for v in std]}


# ===========================================================================
# 3. Luồng chính
# ===========================================================================
def main() -> int:
    args = parse_args()
    if args.calib_size < 1:
        print("Lỗi: --calib-size phải >= 1", file=sys.stderr)
        return 2
    if args.imgsz % 32:
        print(f"Lỗi: --imgsz={args.imgsz} phải là bội số của 32 (stride lớn nhất của YOLO)",
              file=sys.stderr)
        return 2
    if not 100 <= args.calib_size <= 500:
        print(f"[Cảnh báo] calib-size={args.calib_size} nằm ngoài khoảng 100-500 đề bài khuyến nghị.")

    try:
        images = list_images(args.data_dir, args.include_aug)
    except FileNotFoundError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    if len(images) < args.calib_size:
        print(f"Lỗi: thư mục chỉ có {len(images)} ảnh, cần {args.calib_size}.", file=sys.stderr)
        return 1

    # Xáo trộn có seed trên danh sách đã sắp xếp: lấy n ảnh đầu (cộng ảnh dự phòng
    # nếu có ảnh hỏng). Chọn ngẫu nhiên đều phủ được nhiều cảnh / điều kiện sáng
    # khác nhau của sa bàn, kể cả một phần ảnh nền (background) như phân phối thật.
    rng = random.Random(args.seed)
    candidates = images[:]
    rng.shuffle(candidates)
    print(f"Nguồn: {args.data_dir} ({len(images)} ảnh hợp lệ"
          f"{'' if args.include_aug else ', đã bỏ ảnh *_aug###'}) | seed={args.seed}")

    try:
        chosen, skipped = build_calibration_set(candidates, args.calib_size, args.imgsz,
                                                args.normalize, args.output, args.workers)
    except (RuntimeError, OSError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        args.output.unlink(missing_ok=True)  # không để lại file .npy dở dang
        return 1
    except BaseException:  # lỗi bất ngờ / Ctrl+C: vẫn dọn file dở dang rồi báo lỗi gốc
        args.output.unlink(missing_ok=True)
        raise

    stats = channel_stats(args.output)
    dtype = "float32" if args.normalize else "uint8"
    manifest = {
        "output": str(args.output),
        "shape": [len(chosen), args.imgsz, args.imgsz, 3],
        "layout": "NHWC",
        "dtype": dtype,
        "value_range": [0.0, 1.0] if args.normalize else [0, 255],
        "color": "RGB",
        "preprocess": {"letterbox": True, "pad_value": PAD_VALUE, "interpolation": "INTER_LINEAR",
                       "center": True, "scaleup": True},
        "hailo_alls_hint": None if args.normalize else
        "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])",
        "seed": args.seed,
        "source_dir": str(args.data_dir),
        "include_aug": args.include_aug,
        "skipped_corrupted": skipped,
        "channel_stats_0_255": stats,
        "files": [p.name for p in chosen],
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    size_mb = args.output.stat().st_size / 2**20
    print(f"\nĐã lưu {args.output} | shape={tuple(manifest['shape'])} | {dtype} | {size_mb:.1f} MB")
    print(f"Manifest: {manifest_path}")
    print(f"Mean RGB (0-255): {stats['mean_rgb']} | Std: {stats['std_rgb']}")
    if skipped:
        print(f"[Cảnh báo] Bỏ qua {len(skipped)} ảnh hỏng (đã thay bằng ảnh khác):")
        for line in skipped[:10]:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
