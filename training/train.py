"""Huấn luyện mô hình Baseline YOLO26 (m/l) cho xe tự hành RC trên GPU RTX 5070 Ti.

Luồng chạy tổng quát (hàm ``main``):
    1. Đọc tham số dòng lệnh, nạp ``configs/hyp.yaml`` và ``configs/dataset.yaml``.
    2. Phân giải đường dẫn dataset thành đường dẫn tuyệt đối (script chạy được từ
       bất kỳ thư mục nào) và kiểm tra ràng buộc "không lật ảnh".
    3. Bật TensorBoard logging của Ultralytics.
    4. Nạp ``yolo26m.pt`` (hoặc ``yolo26l.pt``), in tổng số Parameters và FLOPs
       (cả đồ thị lúc huấn luyện lẫn đồ thị NMS-free sẽ triển khai lên Hailo-8).
    5. Gắn callback Early Stopping giám sát mAP@0.5:0.95 trên Val và callback lưu
       ``best_fp32.pt`` (trọng số FP32 thật của epoch tốt nhất).
    6. Huấn luyện; Ultralytics tự lưu ``best.pt`` / ``last.pt`` (FP16).
    7. In tóm tắt: mAP tốt nhất, kích thước file, độ phức tạp mô hình tốt nhất.

Yêu cầu môi trường (server):
    - ultralytics >= 8.4 (phiên bản có YOLO26; đã kiểm thử với 8.4.174), tensorboard.
    - RTX 5070 Ti thuộc kiến trúc Blackwell (sm_120) -> cần PyTorch build với
      CUDA >= 12.8 (ví dụ torch >= 2.7 bản cu128), nếu không sẽ báo lỗi
      "no kernel image is available for execution on the device".

Ví dụ:
    python training/train.py
    python training/train.py --model yolo26l.pt --batch 12 --epochs 300
    python training/train.py --batch -1          # AutoBatch: tự dò batch theo VRAM
    python training/train.py --resume            # tiếp tục run bị ngắt giữa chừng
    tensorboard --logdir runs/detect             # xem loss/mAP theo epoch
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO, settings
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

# ---------------------------------------------------------------------------
# Đường dẫn mặc định: tính từ vị trí file này nên chạy từ đâu cũng đúng.
# ---------------------------------------------------------------------------
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_DATA_CFG = TRAINING_DIR / "configs" / "dataset.yaml"
DEFAULT_HYP_CFG = TRAINING_DIR / "configs" / "hyp.yaml"
DEFAULT_PROJECT_DIR = PROJECT_ROOT / "runs" / "detect"

# Tên metric mà Ultralytics ghi vào trainer.metrics / results.csv sau mỗi lần
# validate. Hậu tố "(B)" = Box (phân biệt với "(M)" = Mask của segmentation).
MAP50_95_KEY = "metrics/mAP50-95(B)"

# ---------------------------------------------------------------------------
# GHI CHÚ KIẾN TRÚC: YOLO26 là mô hình NMS-free (không dùng Non-Maximum
# Suppression).
# ---------------------------------------------------------------------------
# - Các YOLO trước (v5/v8/v11) dự đoán NHIỀU box chồng lên nhau cho cùng một
#   vật thể (gán nhãn one-to-many), rồi phải chạy NMS ở hậu xử lý để giữ lại
#   box tốt nhất. NMS là thuật toán tuần tự, rẽ nhánh nhiều, số phần tử động ->
#   KHÔNG chạy được trên NPU, phải đẩy về CPU của Raspberry Pi 5.
# - Head của YOLO26 có HAI nhánh:
#     * one-to-one : mỗi vật thể chỉ khớp đúng MỘT dự đoán -> đầu ra đã là danh
#                    sách box cuối cùng, KHÔNG cần NMS (end-to-end).
#     * one-to-many: nhánh kiểu YOLO cũ, chỉ dùng làm tín hiệu giám sát phụ lúc
#                    huấn luyện để hội tụ nhanh (trọng số loss giảm dần theo epoch).
#   YOLO26 cũng bỏ DFL (Distribution Focal Loss, reg_max=1) -> đồ thị gọn hơn,
#   ít Softmax, dễ lượng tử hóa INT8 và ánh xạ lên Hailo-8 hơn.
# - LƯU Ý QUAN TRỌNG (Ultralytics 8.4): mặc định ``nms=None`` khiến validate /
#   predict / export dùng nhánh one-to-many + NMS ngoài. Vì ta sẽ triển khai
#   nhánh NMS-free, script đặt ``nms=False`` để mAP dùng cho Early Stopping và
#   chọn best.pt được đo trên ĐÚNG nhánh one-to-one. Ở Giai đoạn 3 cũng phải
#   export với ``model.export(format="onnx", nms=False)`` (export không đọc lại
#   tham số này từ checkpoint).
# - Hệ quả cho Giai đoạn 3: không có khối NMS để cắt khỏi ONNX; chỉ cần dùng
#   Netron xem phần đuôi đồ thị (decode / TopK / Gather) để quyết định phần nào
#   chạy trên NPU, phần nào chạy trên CPU.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh; tham số nào được truyền sẽ GHI ĐÈ hyp.yaml."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện Baseline YOLO26 (m/l) trên RTX 5070 Ti.")
    parser.add_argument("--model", default="yolo26m.pt",
                        help="Trọng số khởi tạo: yolo26m.pt (mặc định) hoặc yolo26l.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_CFG,
                        help="File cấu hình dataset (YOLO format)")
    parser.add_argument("--hyp", type=Path, default=DEFAULT_HYP_CFG,
                        help="File siêu tham số huấn luyện")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Số epoch tối đa (Early Stopping thường dừng sớm hơn)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Số epoch liên tiếp mAP@0.5:0.95 (Val) không cải thiện thì dừng")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Mức tăng mAP@0.5:0.95 tối thiểu để tính là 'cải thiện'")
    parser.add_argument("--batch", type=int, default=None,
                        help="Ghi đè batch trong hyp.yaml; -1 = AutoBatch (~60%% VRAM)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Ghi đè imgsz trong hyp.yaml (bội số của 32)")
    parser.add_argument("--device", default="0",
                        help="GPU id ('0'), nhiều GPU ('0,1') hoặc 'cpu'")
    parser.add_argument("--workers", type=int, default=8,
                        help="Số tiến trình DataLoader")
    parser.add_argument("--cache", choices=["false", "ram", "disk"], default="false",
                        help="Cache ảnh để tăng tốc đọc dữ liệu (ram tốn ~10 GB RAM)")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_DIR,
                        help="Thư mục gốc chứa các lần chạy")
    parser.add_argument("--name", default=None,
                        help="Tên lần chạy (mặc định: <model>_baseline). Lưu ý: chạy "
                             "lại cùng tên sẽ tạo thư mục <name>2, <name>3...")
    parser.add_argument("--seed", type=int, default=42, help="Seed để tái lập kết quả")
    parser.add_argument("--resume", action="store_true",
                        help="Tiếp tục run --name bị NGẮT giữa chừng (từ weights/last.pt)")
    return parser.parse_args()


# ===========================================================================
# 1. Nạp & kiểm tra cấu hình
# ===========================================================================
def load_yaml(path: Path) -> dict[str, Any]:
    """Đọc file YAML thành dict (UTF-8 vì file có comment tiếng Việt)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_dataset_yaml(data_cfg: Path, out_file: Path) -> Path:
    """Tạo bản sao dataset.yaml với ``path`` TUYỆT ĐỐI để chạy được từ mọi thư mục.

    Ultralytics hiểu ``path`` tương đối theo thư mục hiện hành (CWD), nên
    ``path: ../data/processed`` chỉ đúng khi chạy từ ``training/``. Hàm này thử
    lần lượt: thư mục training/ (đúng với quy ước trong dataset.yaml), CWD và
    thư mục chứa file yaml; chọn ứng viên đầu tiên có thư mục ``train``.
    """
    cfg = load_yaml(data_cfg)
    raw_path = Path(cfg.get("path", "."))
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [TRAINING_DIR / raw_path, Path.cwd() / raw_path,
                      data_cfg.resolve().parent / raw_path]

    for root in candidates:
        if (root / cfg["train"]).is_dir():
            cfg["path"] = root.resolve().as_posix()
            break
    else:
        tried = "\n  ".join(str(c.resolve()) for c in candidates)
        raise FileNotFoundError(
            f"Không tìm thấy thư mục '{cfg['train']}' của dataset. Đã thử:\n  {tried}\n"
            "Hãy chạy data_engineering/clean_and_filter.py trước.")

    # Kiểm tra nhanh val/test có tồn tại (thiếu Val thì Early Stopping vô nghĩa).
    for split in ("val", "test"):
        if split in cfg and not (Path(cfg["path"]) / cfg[split]).is_dir():
            raise FileNotFoundError(f"Thiếu thư mục {split}: {cfg['path']}/{cfg[split]}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out_file


def build_train_args(args: argparse.Namespace, hyp: dict[str, Any],
                     data_yaml: Path, run_name: str) -> dict[str, Any]:
    """Gộp hyp.yaml + tham số dòng lệnh thành bộ đối số cho ``model.train()``.

    Lý do chọn các giá trị (đặt trong hyp.yaml):
      - optimizer=AdamW, lr0=0.001: AdamW tự thích nghi bước học cho từng tham
        số, hội tụ ổn định khi fine-tune từ trọng số COCO sang 10 lớp sa bàn;
        lr0 nhỏ (0.001) vì 0.01 (mặc định cho SGD) làm AdamW dao động mạnh.
        Khai báo optimizer tường minh nên Ultralytics KHÔNG tự chọn MuSGD
        (lựa chọn của chế độ optimizer='auto' với YOLO26).
      - warmup_bias_lr=0.0: với Adam/AdamW, lr của bias khi warmup không nên
        vượt 0.01 (Ultralytics chỉ tự hạ khi optimizer='auto').
      - cos_lr=True: Cosine Annealing giảm lr mượt từ lr0 về lr0*lrf, các epoch
        cuối học "tinh" giúp mAP@0.5:0.95 (đòi hỏi box chính xác) tăng thêm.
      - batch=16 @ 640: cân bằng giữa VRAM 16 GB của RTX 5070 Ti và việc GPU là
        tài nguyên dùng chung trên server; --batch -1 để AutoBatch tự dò.
      - fliplr=flipud=0: lật ảnh làm đổi nghĩa biển một chiều/vòng xuyến.
    """
    train_args: dict[str, Any] = dict(hyp)

    # Tham số dòng lệnh (nếu có) được ưu tiên hơn hyp.yaml.
    if args.batch is not None:
        train_args["batch"] = args.batch
    if args.imgsz is not None:
        train_args["imgsz"] = args.imgsz

    # Ràng buộc bắt buộc: tuyệt đối không lật ảnh (biển chỉ hướng bị sai nghĩa).
    for key in ("fliplr", "flipud"):
        if float(train_args.get(key, 0.0)) != 0.0:
            raise ValueError(f"{key}={train_args[key]} bị cấm: lật ảnh làm hỏng "
                             f"ngữ nghĩa biển chỉ hướng. Đặt {key}: 0.0 trong hyp.yaml.")
        train_args[key] = 0.0

    train_args.update(
        data=str(data_yaml),
        epochs=args.epochs,
        # Early Stopping nội bộ của Ultralytics theo "fitness". Từ bản 8.4,
        # fitness = 100% mAP@0.5:0.95 (các bản cũ trộn 10% mAP@0.5), nên đặt
        # cùng patience để hai cơ chế nhất quán. Callback ở mục 3 giám sát
        # mAP@0.5:0.95 tường minh, không phụ thuộc phiên bản.
        patience=args.patience,
        # YOLO26: validate bằng nhánh one-to-one NMS-free (nhánh sẽ triển khai),
        # thay vì nhánh one-to-many + NMS mặc định (nms=None).
        nms=False,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        project=str(args.project),
        name=run_name,
        exist_ok=False,
        seed=args.seed,
        # Cố gắng tái lập kết quả. Ultralytics bật chế độ warn_only: thao tác
        # CUDA không tất định chỉ cảnh báo, và cache='ram' cũng có thể phá tính
        # tái lập -> sai khác nhỏ giữa các lần chạy vẫn có thể xảy ra.
        deterministic=True,
        amp=True,                # Mixed precision FP16: nhanh hơn, tiết kiệm VRAM
        val=True,                # BẮT BUỘC validate mỗi epoch để Early Stopping hoạt động
        save=True,               # lưu best.pt (mAP tốt nhất) và last.pt (để resume)
        plots=True,              # lưu đường cong loss/mAP, PR curve cho báo cáo
    )
    return train_args


# ===========================================================================
# 2. Thống kê độ phức tạp (Parameters / FLOPs)
# ===========================================================================
def count_parameters(model: torch.nn.Module) -> int:
    """Tổng số tham số (kể cả tham số không cần gradient)."""
    return sum(p.numel() for p in model.parameters())


def deploy_view(model: torch.nn.Module) -> torch.nn.Module:
    """Bản sao "triển khai" của mô hình: fused + chỉ giữ nhánh one-to-one.

    Lúc huấn luyện, head YOLO26 chạy cả hai nhánh one-to-one và one-to-many;
    ``fuse()`` gộp Conv+BN và bỏ nhánh không dùng khi ``end2end=True``. Đây
    chính là đồ thị sẽ export ONNX (nms=False) và biên dịch lên Hailo-8, nên số
    Parameters/FLOPs của nó mới là mốc so sánh cho PTQ / NPU mapping.
    """
    deployed = copy.deepcopy(model).float().eval()
    if hasattr(deployed, "end2end"):
        deployed.end2end = True
    return deployed.fuse(verbose=False)


def print_complexity(title: str, model: torch.nn.Module, imgsz: int) -> None:
    """In Parameters và FLOPs ở kích thước ``imgsz``, cho đồ thị huấn luyện và triển khai.

    ``get_flops`` của Ultralytics đo bằng thư viện thop, trả về GFLOPs với quy
    ước 1 MAC (nhân-cộng) = 2 FLOPs, nên GMACs = GFLOPs / 2. Số MACs hữu ích khi
    so với năng lực NPU (Hailo-8: 26 TOPS INT8 ~ 13 T MAC/s lý thuyết).
    """
    print(f"\n[Complexity] {title}")
    views = (("Huấn luyện (2 nhánh head)", model),
             ("Triển khai (fused, NMS-free)", deploy_view(model)))
    for label, m in views:
        n_params = count_parameters(m)
        gflops = get_flops(m, imgsz)
        flops_txt = (f"{gflops:.2f} GFLOPs (~{gflops / 2:.2f} GMACs) @ {imgsz}x{imgsz}"
                     if gflops else "không đo được (thiếu thop hoặc lỗi khi trace)")
        print(f"  {label:<30}: {n_params:>12,} params ({n_params / 1e6:.2f} M) | {flops_txt}")


def report_initial_complexity(model: YOLO, nc: int, imgsz: int) -> None:
    """In độ phức tạp của mô hình pretrained VÀ của mô hình thực sự được huấn luyện.

    yolo26m.pt được huấn luyện trên COCO (80 lớp). Khi train với dataset 10 lớp,
    Ultralytics dựng lại lớp phân loại của head với nc=10, nên số tham số/FLOPs
    thực tế nhỏ hơn một chút. ``deepcopy`` cấu hình vì DetectionModel sửa dict
    yaml tại chỗ (ghi đè nc của mô hình pretrained).
    """
    print_complexity(f"{Path(model.ckpt_path or 'model').name} "
                     f"(pretrained COCO, nc={model.model.nc})", model.model, imgsz)
    target = DetectionModel(cfg=copy.deepcopy(model.model.yaml), nc=nc, verbose=False)
    print_complexity(f"Kiến trúc sẽ huấn luyện (nc={nc})", target, imgsz)


# ===========================================================================
# 3. Early Stopping theo mAP@0.5:0.95 (Validation)
# ===========================================================================
class MAP5095EarlyStopping:
    """Callback dừng huấn luyện khi mAP@0.5:0.95 trên Val không cải thiện.

    Cách hoạt động (gắn vào 3 sự kiện của Ultralytics trainer):
      - ``on_train_start`` -> ``seed``: khi resume, khôi phục best/bộ đếm từ
        trạng thái EarlyStopping lưu trong checkpoint (không đếm lại từ 0).
      - ``on_train_epoch_start`` -> ``arm``: đánh dấu đang ở một epoch THẬT.
      - ``on_fit_epoch_end`` -> ``__call__``: chạy sau khi đã validate và lưu
        checkpoint. Đọc ``trainer.metrics["metrics/mAP50-95(B)"]``:
          * > best + min_delta: ghi nhận best mới, reset bộ đếm;
          * ngược lại: tăng bộ đếm; khi đạt ``patience`` thì đặt
            ``trainer.stop = True`` -> vòng lặp thoát ở cuối epoch này.
    Cờ ``armed`` cần thiết vì sau khi huấn luyện xong, ``final_eval`` của
    Ultralytics gọi lại ``on_fit_epoch_end`` để log kết quả best.pt -> nếu không
    chặn sẽ đếm thêm một "epoch ma". Với DDP nhiều GPU, các tiến trình rank > 0
    có ``trainer.metrics = None`` nên được bỏ qua; quyết định dừng của rank 0
    được Ultralytics broadcast sang các rank khác.
    Vì sao vẫn cần callback dù Ultralytics có ``patience``? Để giám sát ĐÚNG
    mAP@0.5:0.95 bất kể phiên bản (bản cũ dùng fitness trộn mAP@0.5), có thêm
    ``min_delta`` và log rõ ràng từng epoch cho báo cáo.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_map: float | None = None
        self.best_epoch = -1
        self.bad_epochs = 0
        self.armed = False

    def seed(self, trainer: Any) -> None:
        """Khôi phục trạng thái khi resume (fitness = mAP@0.5:0.95 từ Ultralytics 8.4)."""
        if trainer.start_epoch > 0 and trainer.stopper.best_fitness:
            self.best_map = float(trainer.stopper.best_fitness)
            self.best_epoch = int(trainer.stopper.best_epoch)
            self.bad_epochs = trainer.start_epoch - self.best_epoch

    def arm(self, trainer: Any) -> None:
        """Chỉ epoch huấn luyện thật mới được tính vào bộ đếm."""
        self.armed = True

    def __call__(self, trainer: Any) -> None:
        if not self.armed:  # lần gọi từ final_eval -> bỏ qua
            return
        self.armed = False
        current = (trainer.metrics or {}).get(MAP50_95_KEY)
        if current is None:  # rank > 0 (DDP) hoặc epoch không validate
            return

        epoch = trainer.epoch + 1  # trainer.epoch đếm từ 0
        if self.best_map is None or current > self.best_map + self.min_delta:
            self.best_map, self.best_epoch, self.bad_epochs = float(current), epoch, 0
        else:
            self.bad_epochs += 1

        print(f"[EarlyStopping] epoch {epoch}: mAP50-95={current:.4f} | "
              f"best={self.best_map:.4f} (epoch {self.best_epoch}) | "
              f"không cải thiện {self.bad_epochs}/{self.patience}")
        if self.bad_epochs >= self.patience:
            print(f"[EarlyStopping] Dừng sớm: {self.patience} epoch liên tiếp "
                  "mAP@0.5:0.95 (Val) không cải thiện.")
            trainer.stop = True


# ===========================================================================
# 4. TensorBoard & checkpoint FP32
# ===========================================================================
def enable_tensorboard() -> None:
    """Bật callback TensorBoard tích hợp sẵn của Ultralytics.

    Phải gọi TRƯỚC ``model.train()`` vì trainer chỉ đăng ký callback lúc khởi tạo.
    Lưu ý: ``settings.update`` ghi vĩnh viễn vào settings.json toàn cục của
    Ultralytics trên máy. Log (loss train/val, mAP, P, R, learning rate theo
    epoch) nằm trong thư mục của lần chạy: ``tensorboard --logdir runs/detect``.
    """
    if importlib.util.find_spec("tensorboard") is None:
        print("[Cảnh báo] Chưa cài tensorboard -> sẽ không có log TensorBoard. "
              "Cài bằng: pip install tensorboard")
    settings.update({"tensorboard": True})


def save_best_fp32(trainer: Any) -> None:
    """Callback ``on_model_save``: lưu ``best_fp32.pt`` khi epoch hiện tại là best.

    Ultralytics chỉ lưu checkpoint ở FP16 (``best.pt``/``last.pt`` gọi ``.half()``),
    nên ép kiểu best.pt lên FP32 sau đó KHÔNG lấy lại được độ chính xác đã mất.
    Ở đây ta chụp trực tiếp trọng số EMA FP32 trong bộ nhớ đúng lúc Ultralytics
    lưu best.pt -> mốc gốc không qua làm tròn FP16 cho ONNX export và PTQ INT8 ở
    Giai đoạn 3 (tránh lượng tử hóa chồng lên sai số FP16).
    """
    if trainer.fitness is None or trainer.best_fitness != trainer.fitness:
        return
    ema = copy.deepcopy(trainer.ema.ema).float()
    ema.criterion = None  # bỏ hàm loss (không cần khi suy luận, gây lỗi pickle)
    torch.save({"model": ema, "train_args": vars(trainer.args), "epoch": -1},
               trainer.wdir / "best_fp32.pt")


def best_map_from_results(save_dir: Path) -> tuple[float, int] | None:
    """Đọc mAP@0.5:0.95 Val tốt nhất từ ``results.csv`` (đúng cả khi resume/DDP)."""
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        return None
    with open(results_csv, encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    rows = [r for r in rows if r.get(MAP50_95_KEY, "").strip()]
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r[MAP50_95_KEY]))
    return float(best[MAP50_95_KEY]), int(float(best["epoch"]))


def summarize_run(trainer: Any) -> None:
    """In tóm tắt sau huấn luyện: vị trí trọng số, mAP tốt nhất, độ phức tạp."""
    save_dir = Path(trainer.save_dir)
    print("\n" + "=" * 78)
    print(f"Thư mục kết quả : {save_dir}")
    best = best_map_from_results(save_dir)
    if best:
        print(f"mAP@0.5:0.95 Val tốt nhất: {best[0]:.4f} (epoch {best[1]})")

    weights = [save_dir / "weights" / name for name in ("best.pt", "best_fp32.pt")]
    for path in weights:
        if path.exists():
            print(f"  {path.name:<13}: {path.stat().st_size / 2**20:7.2f} MB  ({path})")
        else:
            print(f"  [Cảnh báo] Không tìm thấy {path}")
    if weights[0].exists():
        print_complexity("best.pt (mô hình Baseline đã huấn luyện)",
                         YOLO(str(weights[0])).model.float(), int(trainer.args.imgsz))
    print("=" * 78)


# ===========================================================================
# 5. Luồng chính
# ===========================================================================
def load_resumable_checkpoint(last_pt: Path) -> YOLO:
    """Nạp last.pt để resume; báo lỗi rõ ràng nếu checkpoint không resume được.

    Run đã hoàn tất (hoặc đã dừng sớm) có last.pt bị "strip" (xóa optimizer,
    epoch=-1). Khi đó Ultralytics KHÔNG báo lỗi mà lặng lẽ bắt đầu một lần train
    MỚI trên coco8.yaml với siêu tham số mặc định (kể cả fliplr=0.5) -> chặn ở đây.
    """
    if not last_pt.exists():
        raise SystemExit(f"Không có {last_pt} để resume (kiểm tra --name: chạy lại "
                         "cùng tên sẽ tạo thư mục <name>2, <name>3...).")
    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    if ckpt.get("epoch", -1) < 0 or ckpt.get("optimizer") is None:
        raise SystemExit(f"{last_pt} thuộc run đã hoàn tất/đã dừng sớm -> không resume "
                         "được. Hãy chạy một lần huấn luyện mới.")
    print("Resume với tham số đã lưu:", {k: ckpt["train_args"].get(k) for k in (
        "data", "imgsz", "batch", "optimizer", "lr0", "epochs", "patience")})
    return YOLO(str(last_pt))


def main() -> None:
    args = parse_args()
    run_name = args.name or f"{Path(args.model).stem}_baseline"
    device = ("CPU" if args.device == "cpu" or not torch.cuda.is_available()
              else torch.cuda.get_device_name(int(str(args.device).split(",")[0])))
    print(f"Thiết bị  : {device}")

    # Bước 3: bật TensorBoard trước khi trainer được tạo.
    enable_tensorboard()

    # Bước 5 (khởi tạo): callback Early Stopping theo mAP@0.5:0.95 + lưu FP32.
    stopper = MAP5095EarlyStopping(args.patience, args.min_delta)
    callbacks = {
        "on_train_start": stopper.seed,
        "on_train_epoch_start": stopper.arm,
        "on_fit_epoch_end": stopper,
        "on_model_save": save_best_fp32,
    }

    if args.resume:
        # Resume: Ultralytics dùng lại TOÀN BỘ tham số lưu trong last.pt.
        model = load_resumable_checkpoint(args.project / run_name / "weights" / "last.pt")
        for event, fn in callbacks.items():
            model.add_callback(event, fn)
        model.train(resume=True)
    else:
        # Bước 1-2: nạp cấu hình, phân giải dataset, gộp siêu tham số.
        hyp = load_yaml(args.hyp)
        data_yaml = resolve_dataset_yaml(args.data,
                                         args.project / f"{run_name}_dataset.yaml")
        data_cfg = load_yaml(data_yaml)
        nc = int(data_cfg.get("nc") or len(data_cfg["names"]))
        train_args = build_train_args(args, hyp, data_yaml, run_name)
        imgsz = int(train_args.get("imgsz", 640))
        print(f"Dataset   : {data_yaml} (nc={nc})")
        print("Tham số   :", {k: train_args[k] for k in (
            "imgsz", "batch", "optimizer", "lr0", "cos_lr", "epochs", "patience", "nms")
            if k in train_args})

        # Bước 4: nạp mô hình và in Parameters / FLOPs.
        model = YOLO(args.model)
        report_initial_complexity(model, nc, imgsz)
        for event, fn in callbacks.items():
            model.add_callback(event, fn)

        # Bước 6: huấn luyện.
        model.train(**train_args)

    # Bước 7: tóm tắt kết quả.
    summarize_run(model.trainer)

    # Giải phóng VRAM: GPU là tài nguyên dùng chung trên server.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
