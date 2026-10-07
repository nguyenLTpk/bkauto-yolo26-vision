"""Đánh giá mô hình Baseline YOLO26 (FP32/FP16) trên tập Test.

Luồng chạy (hàm ``main``):
    1. Tìm trọng số tốt nhất: ưu tiên ``best_fp32.pt`` (FP32 thật do train.py lưu),
       nếu không có thì dùng ``best.pt`` (FP16) của lần chạy mới nhất trong
       ``runs/detect``; hoặc chỉ định trực tiếp bằng ``--weights``.
    2. Đánh giá trên split='test' với ``nms=False`` (nhánh one-to-one NMS-free của
       YOLO26 - đúng nhánh sẽ triển khai lên Hailo-8, giống lúc train).
    3. In bảng: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 toàn cục và từng lớp,
       kèm bảng tổng hợp theo 2 nhóm của đề bài (traffic_sign / pedestrian).
    4. Lượt thứ hai ở ngưỡng confidence triển khai (mặc định 0.25) để dựng
       Ma trận nhầm lẫn có ý nghĩa, lưu ảnh + CSV.
    5. Thống kê độ phức tạp (kích thước file, Parameters, FLOPs/MACs) làm mốc so
       sánh cho PTQ INT8 và NPU mapping ở Giai đoạn 3.
    6. Ghi toàn bộ số liệu ra ``reports/baseline_eval/<run>/`` (JSON + CSV + ảnh).

Ví dụ:
    python training/evaluate_baseline.py
    python training/evaluate_baseline.py --weights runs/detect/yolo26m_baseline/weights/best.pt
    python training/evaluate_baseline.py --fp16 --cm-conf 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils.torch_utils import get_flops

# Dùng lại các hàm của train.py (cùng thư mục training/) để hai script nhất quán:
# cùng cách phân giải dataset.yaml và cùng định nghĩa "đồ thị triển khai".
from train import (DEFAULT_DATA_CFG, DEFAULT_PROJECT_DIR, PROJECT_ROOT,
                   count_parameters, deploy_view, load_yaml, resolve_dataset_yaml)

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "baseline_eval"

# Hai nhóm lớp theo đề bài. Dataset giữ nguyên 10 class gốc (0..8 là các loại
# biển báo, 9 là người đi bộ), nên chỉ số nhóm được tổng hợp từ chỉ số từng lớp.
CLASS_GROUPS = {"traffic_sign": list(range(9)), "pedestrian": [9]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá Baseline YOLO26 trên tập Test.")
    parser.add_argument("--weights", type=Path, default=None,
                        help="File trọng số; mặc định tự tìm trong runs/detect")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_PROJECT_DIR,
                        help="Thư mục chứa các lần huấn luyện")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_CFG,
                        help="File cấu hình dataset")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Tập đánh giá (mặc định test)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Kích thước ảnh; mặc định lấy imgsz lúc train")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU id hoặc 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true",
                        help="Suy luận FP16 (mặc định FP32)")
    parser.add_argument("--cm-conf", type=float, default=0.25,
                        help="Ngưỡng confidence để dựng ma trận nhầm lẫn")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR,
                        help="Thư mục lưu báo cáo")
    return parser.parse_args()


# ===========================================================================
# 1. Tìm trọng số
# ===========================================================================
def find_best_weights(runs_dir: Path) -> Path:
    """Chọn trọng số của lần huấn luyện mới nhất, ưu tiên best_fp32.pt.

    best_fp32.pt là trọng số EMA FP32 chụp ngay lúc lưu best.pt (xem train.py),
    nên là mốc "gốc" chính xác nhất; best.pt đã bị Ultralytics làm tròn về FP16.
    """
    run_dirs = sorted((p.parent.parent for p in runs_dir.glob("*/weights/best*.pt")),
                      key=lambda d: max(f.stat().st_mtime for f in d.glob("weights/best*.pt")))
    if not run_dirs:
        raise SystemExit(f"Không tìm thấy best.pt / best_fp32.pt trong {runs_dir}/*/weights. "
                         "Hãy chạy training/train.py trước hoặc dùng --weights.")
    weights_dir = run_dirs[-1] / "weights"
    for name in ("best_fp32.pt", "best.pt"):
        if (weights_dir / name).exists():
            return weights_dir / name
    raise SystemExit(f"Không có trọng số trong {weights_dir}")  # không xảy ra


def run_name_of(weights: Path) -> str:
    """Tên lần chạy = thư mục cha của weights/ (vd. yolo26m_baseline)."""
    return weights.parent.parent.name if weights.parent.name == "weights" else weights.stem


# ===========================================================================
# 2. Đánh giá
# ===========================================================================
def precision_kwargs(fp16: bool) -> dict[str, Any]:
    """Tham số chọn độ chính xác suy luận, tương thích nhiều phiên bản Ultralytics.

    Từ 8.4.x, tham số ``half`` được thay bằng ``quantize`` (16 = FP16, None = FP32).
    """
    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": 16 if fp16 else None}
    return {"half": fp16}


def evaluate(model: YOLO, data_yaml: Path, args: argparse.Namespace, imgsz: int,
             save_dir: Path, conf: float | None, plots: bool) -> Any:
    """Gọi ``model.val`` một lần và trả về đối tượng DetMetrics.

    - ``nms=False``: dùng nhánh one-to-one NMS-free của YOLO26, giống lúc train.
    - ``conf=None`` -> Ultralytics dùng 0.001: ngưỡng thấp để đường
      Precision-Recall phủ toàn dải recall, đây là quy ước chuẩn khi tính mAP.
    - ``rect=False`` + letterbox vuông ``imgsz``: giống đầu vào cố định của NPU.
    """
    return model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        nms=False,
        conf=conf,
        rect=False,
        plots=plots,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        verbose=False,
        **precision_kwargs(args.fp16),
    )


# ===========================================================================
# 3. Trích xuất chỉ số
# ===========================================================================
def f1_score(p: float, r: float) -> float:
    """F1 = trung bình điều hòa của Precision và Recall."""
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def count_split_images(data_yaml: Path, split: str) -> int:
    """Đếm số ảnh của split (kể cả ảnh nền không có nhãn) từ dataset.yaml."""
    cfg = load_yaml(data_yaml)
    split_dir = Path(cfg["path"]) / cfg[split]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in split_dir.iterdir() if p.suffix.lower() in exts)


def extract_metrics(metrics: Any, names: dict[int, str], n_images: int) -> dict[str, Any]:
    """Rút chỉ số toàn cục + từng lớp từ DetMetrics của Ultralytics.

    Cách Ultralytics tính (quan trọng khi viết báo cáo):
      - AP mỗi lớp = diện tích dưới đường P-R (nội suy 101 điểm kiểu COCO);
        mAP@0.5:0.95 = trung bình AP trên 10 ngưỡng IoU 0.50, 0.55, ..., 0.95.
      - P, R, F1 từng lớp lấy tại MỘT ngưỡng confidence chung - ngưỡng làm F1
        trung bình các lớp lớn nhất - nên chúng là điểm làm việc "tốt nhất" của
        mô hình chứ không phải tại conf=0.001 dùng để tính mAP.
      - Chỉ các lớp có nhãn trong tập đánh giá (``ap_class_index``) mới có chỉ
        số; lớp vắng mặt được đánh dấu None thay vì 0 để không kéo lệch trung bình.
    """
    box = metrics.box
    per_class: dict[int, dict[str, Any]] = {}
    for i, cls_id in enumerate(box.ap_class_index):
        cls_id = int(cls_id)
        p, r, ap50, ap = box.class_result(i)
        per_class[cls_id] = {
            "name": names[cls_id],
            "images": int(metrics.nt_per_image[cls_id]),
            "instances": int(metrics.nt_per_class[cls_id]),
            "precision": float(p),
            "recall": float(r),
            "f1": float(box.f1[i]),
            "mAP50": float(ap50),
            "mAP50_95": float(ap),
        }

    mp, mr = float(box.mp), float(box.mr)
    overall = {
        "images": n_images,
        "instances": int(metrics.nt_per_class.sum()),
        "precision": mp,
        "recall": mr,
        # F1 toàn cục tính từ P, R trung bình (macro) - cùng quy ước với Ultralytics.
        "f1": f1_score(mp, mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }
    return {"overall": overall, "per_class": per_class,
            "missing_classes": [c for c in names if c not in per_class]}


def group_metrics(per_class: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Tổng hợp theo 2 nhóm của đề bài: trung bình macro các lớp trong nhóm.

    Ví dụ mAP(traffic_sign) = trung bình mAP của 9 loại biển báo (0..8). Đây là
    macro-average theo lớp (mỗi loại biển báo có trọng số như nhau), KHÔNG phải
    mAP của một lớp gộp "traffic_sign" - vì mô hình vẫn phân biệt 9 loại biển.
    """
    groups: dict[str, dict[str, Any]] = {}
    keys = ("precision", "recall", "f1", "mAP50", "mAP50_95")
    for group, ids in CLASS_GROUPS.items():
        rows = [per_class[c] for c in ids if c in per_class]
        if not rows:
            continue
        groups[group] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        groups[group]["instances"] = sum(r["instances"] for r in rows)
        groups[group]["n_classes"] = len(rows)
    return groups


# ===========================================================================
# 4. Độ phức tạp tính toán
# ===========================================================================
def complexity_stats(model: YOLO, weights: Path, imgsz: int) -> dict[str, Any]:
    """Kích thước file, Parameters, FLOPs/MACs cho đồ thị huấn luyện và triển khai.

    - GFLOPs đo bằng thop (qua ``get_flops``), quy ước 1 MAC = 2 FLOPs.
    - "deploy" = fused Conv+BN, chỉ nhánh one-to-one NMS-free: đúng đồ thị sẽ
      export ONNX (nms=False) và biên dịch lên Hailo-8 -> mốc cho Giai đoạn 3.
    """
    files = {p.name: round(p.stat().st_size / 2**20, 2)
             for p in sorted(weights.parent.glob("best*.pt"))}
    stats: dict[str, Any] = {"imgsz": imgsz, "weight_files_MB": files}
    net = model.model.float()
    for label, m in (("train_graph", net), ("deploy_graph", deploy_view(net))):
        gflops = get_flops(m, imgsz) or None
        stats[label] = {
            "params": count_parameters(m),
            "GFLOPs": round(gflops, 2) if gflops else None,
            "GMACs": round(gflops / 2, 2) if gflops else None,
        }
    return stats


# ===========================================================================
# 5. In bảng & lưu báo cáo
# ===========================================================================
def fmt(v: Any, pct: bool = True) -> str:
    """Định dạng số: tỉ lệ -> %, None -> '-'."""
    if v is None:
        return "-"
    return f"{v * 100:6.2f}" if pct else f"{v:,}"


def print_report(res: dict[str, Any], groups: dict[str, Any], cx: dict[str, Any],
                 speed: dict[str, float], header: str) -> None:
    """In bảng kết quả ra console (các chỉ số tính bằng %)."""
    line = "=" * 100
    cols = f"{'Class':<28}{'Images':>8}{'Inst.':>8}{'P':>9}{'R':>9}{'F1':>9}" \
           f"{'mAP50':>10}{'mAP50-95':>11}"
    print(f"\n{line}\n{header}\n{line}\n{cols}\n{'-' * 100}")

    o = res["overall"]
    print(f"{'all':<28}{fmt(o['images'], False):>8}{fmt(o['instances'], False):>8}"
          f"{fmt(o['precision']):>9}{fmt(o['recall']):>9}{fmt(o['f1']):>9}"
          f"{fmt(o['mAP50']):>10}{fmt(o['mAP50_95']):>11}")
    for cls_id, c in sorted(res["per_class"].items()):
        print(f"{f'{cls_id}: ' + c['name']:<28}{c['images']:>8,}{c['instances']:>8,}"
              f"{fmt(c['precision']):>9}{fmt(c['recall']):>9}{fmt(c['f1']):>9}"
              f"{fmt(c['mAP50']):>10}{fmt(c['mAP50_95']):>11}")
    for cls_id in res["missing_classes"]:
        print(f"{cls_id}: (không có nhãn trong tập đánh giá)")

    print(f"{'-' * 100}\nTheo nhóm (macro-average các lớp trong nhóm):")
    for g, v in groups.items():
        label = f"{g} ({v['n_classes']} lớp)"
        print(f"{label:<28}{'':>8}{v['instances']:>8,}"
              f"{fmt(v['precision']):>9}{fmt(v['recall']):>9}{fmt(v['f1']):>9}"
              f"{fmt(v['mAP50']):>10}{fmt(v['mAP50_95']):>11}")

    print(f"{'-' * 100}\nĐộ phức tạp @ {cx['imgsz']}x{cx['imgsz']}:")
    for name, mb in cx["weight_files_MB"].items():
        print(f"  {name:<14}: {mb:8.2f} MB")
    for label in ("train_graph", "deploy_graph"):
        s = cx[label]
        print(f"  {label:<14}: {s['params']:>12,} params ({s['params'] / 1e6:.2f} M) | "
              f"{s['GFLOPs']} GFLOPs | {s['GMACs']} GMACs")
    print(f"Tốc độ trên thiết bị đánh giá (ms/ảnh, chỉ tham khảo - KHÔNG phải số đo "
          f"trên Edge): " + ", ".join(f"{k}={v:.2f}" for k, v in speed.items()
                                      if k != "loss"))
    print(line)


def save_confusion_matrix(cm_metrics: Any, cm_dir: Path, out_dir: Path,
                          names: dict[int, str], conf: float) -> list[Path]:
    """Lưu ma trận nhầm lẫn (ảnh thường + chuẩn hóa) và bản CSV số đếm.

    Ma trận có kích thước (nc+1) x (nc+1), hàng = lớp DỰ ĐOÁN, cột = lớp THẬT;
    hàng/cột cuối "background": cột background = báo nhầm (FP), hàng background
    = bỏ sót (FN). Ghép cặp dự đoán - nhãn dùng IoU >= 0.45.
    Ultralytics mặc định dựng ma trận ở conf của lượt tính mAP (0.001) -> cột
    background đầy FP vô nghĩa; vì vậy ta dựng ở ``--cm-conf`` (gần ngưỡng triển
    khai thật) trong một lượt đánh giá riêng.
    """
    saved: list[Path] = []
    tag = f"conf{conf:g}"
    for src_name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = cm_dir / src_name
        if not src.exists():  # phòng khi phiên bản khác không tự vẽ
            cm_metrics.confusion_matrix.plot(normalize="normalized" in src_name,
                                             save_dir=cm_dir)
        if src.exists():
            dst = out_dir / src_name.replace(".png", f"_{tag}.png")
            shutil.copyfile(src, dst)
            saved.append(dst)

    labels = [names[i] for i in sorted(names)] + ["background"]
    csv_path = out_dir / f"confusion_matrix_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted \\ true"] + labels)
        for label, row in zip(labels, cm_metrics.confusion_matrix.matrix):
            writer.writerow([label] + [int(v) for v in row])
    saved.append(csv_path)
    return saved


def save_tables(res: dict[str, Any], groups: dict[str, Any], out_dir: Path) -> Path:
    """Ghi bảng từng lớp + nhóm ra CSV (dễ chuyển sang bảng LaTeX)."""
    csv_path = out_dir / "metrics_per_class.csv"
    fields = ["precision", "recall", "f1", "mAP50", "mAP50_95"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "name", "images", "instances"] + fields)
        o = res["overall"]
        writer.writerow(["all", "all", o["images"], o["instances"]] + [o[k] for k in fields])
        for cls_id, c in sorted(res["per_class"].items()):
            writer.writerow([cls_id, c["name"], c["images"], c["instances"]]
                            + [c[k] for k in fields])
        for g, v in groups.items():
            writer.writerow([f"group:{g}", g, "", v["instances"]] + [v[k] for k in fields])
    return csv_path


# ===========================================================================
# 6. Luồng chính
# ===========================================================================
def main() -> None:
    args = parse_args()

    # Bước 1: tìm trọng số và nạp mô hình.
    weights = (args.weights or find_best_weights(args.runs_dir)).resolve()
    model = YOLO(str(weights))
    run_name = run_name_of(weights)
    imgsz = args.imgsz or int(model.ckpt.get("train_args", {}).get("imgsz", 640))
    out_dir = args.out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = resolve_dataset_yaml(args.data, out_dir / "dataset_resolved.yaml")
    names = {int(k): v for k, v in load_yaml(data_yaml)["names"].items()}
    print(f"Trọng số : {weights}")
    print(f"Dataset  : {data_yaml} | split={args.split} | imgsz={imgsz} | "
          f"{'FP16' if args.fp16 else 'FP32'} | nms=False (NMS-free)")

    # Bước 2: lượt chính tính mAP/P/R/F1 (conf=0.001, kèm đường PR/F1 cho báo cáo).
    metrics = evaluate(model, data_yaml, args, imgsz, out_dir / "val_map",
                       conf=None, plots=True)
    res = extract_metrics(metrics, names, count_split_images(data_yaml, args.split))
    groups = group_metrics(res["per_class"])
    # Ma trận nhầm lẫn của lượt này dựng ở conf=0.001 (đầy FP vô nghĩa) -> xóa
    # để không nhầm với ma trận chính thức ở --cm-conf (Bước 4).
    for stale in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        (out_dir / "val_map" / stale).unlink(missing_ok=True)

    # Bước 4: lượt phụ ở ngưỡng triển khai để dựng ma trận nhầm lẫn.
    cm_dir = out_dir / f"val_cm_conf{args.cm_conf:g}"
    cm_metrics = evaluate(model, data_yaml, args, imgsz, cm_dir,
                          conf=args.cm_conf, plots=True)
    cm_files = save_confusion_matrix(cm_metrics, cm_dir, out_dir, names, args.cm_conf)

    # Bước 5: độ phức tạp tính toán.
    cx = complexity_stats(model, weights, imgsz)

    # Bước 3 + 6: in bảng và lưu báo cáo.
    print_report(res, groups, cx, metrics.speed,
                 f"BASELINE {run_name} | {weights.name} | split={args.split} | "
                 f"{'FP16' if args.fp16 else 'FP32'} | NMS-free (chỉ số tính bằng %)")
    table_csv = save_tables(res, groups, out_dir)
    report = {
        "weights": str(weights), "split": args.split, "imgsz": imgsz,
        "precision_mode": "fp16" if args.fp16 else "fp32", "nms": False,
        "confusion_matrix_conf": args.cm_conf,
        "metrics": res, "groups": groups, "complexity": cx,
        "speed_ms_per_image": metrics.speed,
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() and args.device != "cpu" else "cpu"),
    }
    json_path = out_dir / "baseline_metrics.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nĐã lưu:")
    for p in [json_path, table_csv, *cm_files]:
        print(f"  {p}")
    print(f"  Đường cong PR/F1/P/R: {out_dir / 'val_map'}")


if __name__ == "__main__":
    main()
