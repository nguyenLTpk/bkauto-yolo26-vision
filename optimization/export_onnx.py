"""Xuất YOLO26 (NMS-free) sang ONNX tĩnh, tương thích Hailo Dataflow Compiler (DFC).

Luồng chạy (hàm ``main``):
    1. Xuất PyTorch -> ONNX bằng Ultralytics: batch=1, shape tĩnh 1x3x640x640,
       ``nms=False`` (nhánh one-to-one NMS-free), opset 17, FP32.
    2. Rút gọn đồ thị bằng onnxsim (constant folding, bỏ node thừa); không có
       onnxsim thì dùng onnxslim.
    3. Kiểm tra hợp lệ (``onnx.checker``), xác nhận mọi shape đều tĩnh, in chi
       tiết tên / kiểu / shape của Input & Output -> ``models/yolo26m.onnx``.
    4. Phân tích phần đuôi đồ thị: YOLO26 không có NMS nhưng vẫn có khối decode +
       TopK/GatherElements chọn 300 box - các toán tử này không phù hợp NPU.
       Script tự tìm 6 Conv cuối của head one-to-one (box + class cho 3 tỉ lệ
       stride 8/16/32), lưu tên node làm ``end_node_names`` cho Hailo DFC và
       cắt sẵn đồ thị con thuần NPU -> ``models/yolo26m_npu.onnx``.
    5. Kiểm chứng số học bằng onnxruntime (nếu có):
         a) PyTorch vs ONNX đầy đủ (cùng một ảnh test đã letterbox);
         b) ONNX đầy đủ vs (đồ thị NPU + decode NumPy trên CPU) - xác nhận công
            thức hậu xử lý sẽ dùng trên Raspberry Pi 5 ở Giai đoạn 4 là đúng.
    6. Ghi báo cáo ``models/yolo26m_onnx_report.json`` (op histogram, end nodes,
       công thức decode, kết quả kiểm chứng) phục vụ compile_npu.py và báo cáo.

Yêu cầu: ultralytics >= 8.4, onnx, onnxsim (hoặc onnxslim), onnxruntime (tùy chọn).

Ví dụ:
    python optimization/export_onnx.py
    python optimization/export_onnx.py --weights runs/detect/<run>/weights/best_fp32.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import shape_inference
from ultralytics import YOLO

# Dùng chung hàm letterbox với calibrate_ptq.py (cùng thư mục optimization/) để
# ảnh kiểm thử được tiền xử lý giống hệt dữ liệu hiệu chuẩn PTQ.
from calibrate_ptq import IMAGE_EXTS, letterbox, read_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "yolo26m_baseline-2" / "weights" / "best_fp32.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "yolo26m.onnx"
DEFAULT_TEST_IMAGES = PROJECT_ROOT / "data" / "processed" / "Test" / "images"

# Toán tử điều khiển luồng / chỉ số động: Hailo-8 không ánh xạ được lên NPU
# (hoặc chỉ hỗ trợ rất hạn chế) -> phải nằm ngoài đồ thị biên dịch, chạy trên CPU.
NPU_UNFRIENDLY_OPS = {
    "NonMaxSuppression", "TopK", "GatherElements", "GatherND", "Gather", "ScatterND",
    "NonZero", "Where", "Equal", "Mod", "ReduceMax", "ArgMax", "Cast", "Expand",
    "ConstantOfShape", "Shape", "Range", "Loop", "If",
}

# Conv cuối của head one-to-one, ví dụ "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv":
#   one2one_cv2.<i> = nhánh hồi quy box (4 kênh: l, t, r, b) ở tỉ lệ thứ i
#   one2one_cv3.<i> = nhánh phân loại (nc kênh, logit trước sigmoid) ở tỉ lệ thứ i
# Conv cuối là nn.Conv2d trần (không có "/conv/" như các khối Conv-BN-SiLU).
END_NODE_PATTERN = re.compile(
    r"/one2one_cv(?P<branch>[23])\.(?P<scale>\d+)/one2one_cv[23]\.\d+\.\d+/Conv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xuất YOLO26 sang ONNX tĩnh cho Hailo DFC.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Trọng số PyTorch (khuyến nghị best_fp32.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước đầu vào vuông")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset (17: ổn định với Hailo DFC 3.x)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="File ONNX đầy đủ (end-to-end, đầu ra 1x300x6)")
    parser.add_argument("--simplifier", choices=["onnxsim", "onnxslim", "none"],
                        default="onnxsim", help="Công cụ rút gọn đồ thị")
    parser.add_argument("--no-npu-subgraph", action="store_true",
                        help="Không cắt đồ thị con thuần NPU")
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES,
                        help="Thư mục ảnh dùng để kiểm chứng số học")
    parser.add_argument("--verify-images", type=int, default=16,
                        help="Số ảnh test dùng để kiểm chứng số học")
    parser.add_argument("--no-verify", action="store_true",
                        help="Bỏ qua kiểm chứng bằng onnxruntime")
    return parser.parse_args()


# ===========================================================================
# 1. Xuất ONNX bằng Ultralytics
# ===========================================================================
def export_raw_onnx(weights: Path, imgsz: int, opset: int, work_dir: Path) -> Path:
    """Gọi exporter của Ultralytics với cấu hình tĩnh cho Hailo DFC.

    - ``batch=1, dynamic=False``: mọi trục có kích thước cố định (1x3x640x640).
      DFC biên dịch đồ thị thành luồng dữ liệu (dataflow) cố định trên chip, cấp
      phát bộ nhớ/tài nguyên lúc biên dịch -> không chấp nhận dynamic axes.
    - ``nms=False``: kích hoạt nhánh one-to-one NMS-free của YOLO26 (mặc định
      ``nms=None`` sẽ xuất nhánh one-to-many cần NMS bên ngoài). Không chèn
      toán tử ONNX NonMaxSuppression - Hailo-8 không hỗ trợ trên NPU.
    - ``simplify=False``: Ultralytics mặc định rút gọn bằng onnxslim; ta tự làm
      ở bước sau bằng onnxsim để kiểm soát và ghi log.
    - Xuất trên CPU, FP32: trọng số best_fp32.pt giữ nguyên độ chính xác để PTQ.
    Exporter ghi file cạnh file trọng số, nên ta copy trọng số vào thư mục tạm để
    không làm bẩn thư mục runs/.
    """
    tmp_weights = work_dir / weights.name
    shutil.copyfile(weights, tmp_weights)
    model = YOLO(str(tmp_weights))
    exported = model.export(format="onnx", imgsz=imgsz, batch=1, dynamic=False,
                            opset=opset, simplify=False, nms=False, device="cpu")
    return Path(exported)


def simplify_model(model: onnx.ModelProto, method: str) -> tuple[onnx.ModelProto, str]:
    """Rút gọn đồ thị: gấp hằng số (constant folding), loại node thừa/trùng lặp.

    Với batch=1 tĩnh, các chuỗi Shape -> Gather -> Unsqueeze -> Concat sinh ra
    khi PyTorch trace phép reshape đều được tính sẵn thành hằng số -> đồ thị
    gọn hơn, ít toán tử "động" hơn, parser của DFC dễ xử lý hơn.
    """
    if method == "onnxsim":
        try:
            import onnxsim
            simplified, ok = onnxsim.simplify(model)
            if not ok:
                raise RuntimeError("onnxsim: kiểm tra tương đương sau rút gọn thất bại")
            return simplified, f"onnxsim {onnxsim.__version__}"
        except ImportError:
            print("[Cảnh báo] Chưa cài onnxsim (pip install onnxsim) -> thử onnxslim.")
            method = "onnxslim"
    if method == "onnxslim":
        try:
            import onnxslim
            return onnxslim.slim(model), f"onnxslim {onnxslim.__version__}"
        except ImportError:
            print("[Cảnh báo] Chưa cài onnxslim -> giữ nguyên đồ thị chưa rút gọn.")
    return model, "none"


# ===========================================================================
# 2. Kiểm tra & mô tả đồ thị
# ===========================================================================
def tensor_specs(values: Any) -> list[dict[str, Any]]:
    """Trích tên / kiểu dữ liệu / shape của danh sách ValueInfo (input hoặc output)."""
    specs = []
    for v in values:
        t = v.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in t.shape.dim]
        specs.append({"name": v.name,
                      "dtype": str(onnx.helper.tensor_dtype_to_np_dtype(t.elem_type)),
                      "shape": dims})
    return specs


def validate_model(model: onnx.ModelProto) -> dict[str, list[dict[str, Any]]]:
    """``onnx.checker`` + xác nhận shape tĩnh, batch=1 cho mọi input/output."""
    onnx.checker.check_model(model, full_check=True)
    io = {"inputs": tensor_specs(model.graph.input), "outputs": tensor_specs(model.graph.output)}
    for kind, specs in io.items():
        for spec in specs:
            if not all(isinstance(d, int) and d > 0 for d in spec["shape"]):
                raise ValueError(f"{kind[:-1]} '{spec['name']}' có trục động {spec['shape']} "
                                 "- Hailo DFC yêu cầu đồ thị tĩnh.")
    if io["inputs"][0]["shape"][0] != 1:
        raise ValueError(f"Batch của input phải bằng 1, nhận {io['inputs'][0]['shape']}")
    return io


def print_io(title: str, io: dict[str, list[dict[str, Any]]]) -> None:
    print(f"\n[{title}]")
    for kind, specs in io.items():
        for s in specs:
            print(f"  {kind[:-1]:<6} | {s['name']:<55} | {s['dtype']:<8} | {s['shape']}")


def shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    """Bảng tên tensor -> shape sau khi suy luận shape (shape inference)."""
    inferred = shape_inference.infer_shapes(model)
    values = list(inferred.graph.value_info) + list(inferred.graph.output)
    return {s["name"]: s["shape"] for s in tensor_specs(values)}


def find_npu_end_nodes(model: onnx.ModelProto, imgsz: int) -> list[dict[str, Any]]:
    """Tìm 6 Conv cuối của head one-to-one - ranh giới NPU / CPU.

    Mọi thứ TRƯỚC các Conv này (backbone, neck, head conv) là tính toán dày đặc
    thân thiện NPU. Mọi thứ SAU (decode anchor, sigmoid, TopK, GatherElements)
    là hậu xử lý nhẹ nhưng có chỉ số động -> chạy trên CPU của Raspberry Pi 5.
    """
    shapes = shape_map(model)
    nodes = []
    for node in model.graph.node:
        match = END_NODE_PATTERN.search(node.name) if node.op_type == "Conv" else None
        if not match:
            continue
        shape = shapes.get(node.output[0])
        nodes.append({
            "node": node.name,
            "tensor": node.output[0],
            "kind": "box_ltrb" if match["branch"] == "2" else "cls_logits",
            "scale": int(match["scale"]),
            "shape": shape,
            "stride": imgsz // shape[-1] if shape else None,
        })
    return sorted(nodes, key=lambda n: (n["scale"], n["kind"]))


def tail_analysis(model: onnx.ModelProto, end_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Thống kê toán tử toàn đồ thị và toán tử khó cho NPU nằm SAU điểm cắt."""
    histogram = Counter(n.op_type for n in model.graph.node)
    end_tensors = {n["tensor"] for n in end_nodes}
    # Tập node phía sau điểm cắt = mọi node phụ thuộc (trực tiếp/gián tiếp) vào end tensors.
    downstream: list[onnx.NodeProto] = []
    reached = set(end_tensors)
    for node in model.graph.node:
        if any(i in reached for i in node.input):
            downstream.append(node)
            reached.update(node.output)
    unfriendly_total = {op: c for op, c in histogram.items() if op in NPU_UNFRIENDLY_OPS}
    unfriendly_tail = Counter(n.op_type for n in downstream if n.op_type in NPU_UNFRIENDLY_OPS)
    return {
        "num_nodes": len(model.graph.node),
        "op_histogram": dict(histogram.most_common()),
        "npu_unfriendly_ops_total": unfriendly_total,
        "cpu_tail_nodes": len(downstream),
        "cpu_tail_op_histogram": dict(Counter(n.op_type for n in downstream).most_common()),
        "npu_unfriendly_ops_in_tail": dict(unfriendly_tail),
        "npu_unfriendly_ops_before_cut": {
            op: unfriendly_total[op] - unfriendly_tail.get(op, 0)
            for op in unfriendly_total if unfriendly_total[op] - unfriendly_tail.get(op, 0) > 0},
    }


def extract_npu_subgraph(model: onnx.ModelProto, end_nodes: list[dict[str, Any]],
                         output: Path) -> dict[str, list[dict[str, Any]]]:
    """Cắt đồ thị con từ input 'images' tới 6 Conv cuối -> ONNX thuần NPU."""
    input_names = [i.name for i in model.graph.input]
    sub = onnx.utils.Extractor(shape_inference.infer_shapes(model)).extract_model(
        input_names, [n["tensor"] for n in end_nodes])
    sub.producer_name = "export_onnx.py (NPU subgraph)"
    io = validate_model(sub)
    onnx.save(sub, output)
    return io


# ===========================================================================
# 3. Kiểm chứng số học (onnxruntime)
# ===========================================================================
def load_test_tensors(test_dir: Path, imgsz: int, count: int) -> list[tuple[np.ndarray, str]]:
    """``count`` ảnh test trải đều theo thứ tự tên -> letterbox -> RGB -> NCHW [0,1].

    Không có ảnh test thì dùng một tensor ngẫu nhiên (vẫn kiểm được tính tương
    đương đồ thị, chỉ là ít box vượt ngưỡng).
    """
    images = sorted(p for p in test_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS) \
        if test_dir.is_dir() else []
    if not images:
        rng = np.random.default_rng(0)
        return [(rng.random((1, 3, imgsz, imgsz), dtype=np.float32), "random")]
    picks = [images[i] for i in np.linspace(0, len(images) - 1, min(count, len(images)), dtype=int)]
    tensors = []
    for path in picks:
        image, _, _ = letterbox(read_image(path), imgsz)
        rgb = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        tensors.append((np.ascontiguousarray(rgb), path.name))
    return tensors


def load_torch_reference(weights: Path) -> torch.nn.Module:
    """Mô hình PyTorch FP32, fused, nhánh one-to-one (end2end) - giống đồ thị đã xuất."""
    net = copy.deepcopy(YOLO(str(weights)).model).float().eval()
    if hasattr(net, "end2end"):
        net.end2end = True
    return net.fuse(verbose=False)


def torch_forward(net: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        out = net(torch.from_numpy(x))
    out = out[0] if isinstance(out, (list, tuple)) else out
    return out.numpy()


def decode_npu_outputs(outputs: dict[str, np.ndarray], end_nodes: list[dict[str, Any]],
                       max_det: int = 300) -> np.ndarray:
    """Hậu xử lý CPU cho đầu ra đồ thị NPU, tái hiện head YOLO26 (end2end, reg_max=1).

    Với mỗi tỉ lệ (stride s, lưới H x W):
      - anchor (cx, cy) = (j + 0.5, i + 0.5) theo đơn vị ô lưới;
      - box: x1 = (cx - l)*s, y1 = (cy - t)*s, x2 = (cx + r)*s, y2 = (cy + b)*s
        (l, t, r, b là 4 kênh của Conv box; YOLO26 bỏ DFL nên không cần softmax);
      - score lớp = sigmoid(logit).
    Sau đó lấy top ``max_det`` cặp (anchor, lớp) theo score - tương đương phép
    TopK hai tầng trong đồ thị gốc. KHÔNG cần NMS. Trả về (max_det, 6):
    [x1, y1, x2, y2, score, class_id] theo tọa độ ảnh đã letterbox.
    """
    boxes, scores = [], []
    for scale in sorted({n["scale"] for n in end_nodes}):
        box_node = next(n for n in end_nodes if n["scale"] == scale and n["kind"] == "box_ltrb")
        cls_node = next(n for n in end_nodes if n["scale"] == scale and n["kind"] == "cls_logits")
        ltrb, logits = outputs[box_node["tensor"]][0], outputs[cls_node["tensor"]][0]
        h, w = ltrb.shape[1:]
        stride = box_node["stride"]
        cy, cx = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        l, t, r, b = ltrb
        xyxy = np.stack([cx - l, cy - t, cx + r, cy + b], axis=-1) * stride
        boxes.append(xyxy.reshape(-1, 4))
        sigmoid = 0.5 * (1.0 + np.tanh(0.5 * logits))  # = 1/(1+e^-x), không tràn số
        scores.append(sigmoid.reshape(logits.shape[0], -1).T)
    boxes_all, scores_all = np.concatenate(boxes), np.concatenate(scores)
    nc = scores_all.shape[1]
    flat = scores_all.ravel()
    top = np.argsort(-flat, kind="stable")[:max_det]
    anchor_idx, cls_idx = top // nc, top % nc
    return np.concatenate([boxes_all[anchor_idx], flat[top, None], cls_idx[:, None]], axis=1)


def compare_detections(ref: np.ndarray, test: np.ndarray, conf: float = 0.25) -> dict[str, Any]:
    """So sánh hai tập phát hiện (đã sắp theo score giảm dần) ở ngưỡng ``conf``."""
    ref = ref[ref[:, 4] >= conf]
    test = test[test[:, 4] >= conf]
    k = min(len(ref), len(test))
    box_diff = float(np.abs(ref[:k, :4] - test[:k, :4]).max()) if k else 0.0
    score_diff = float(np.abs(ref[:k, 4] - test[:k, 4]).max()) if k else 0.0
    class_match = bool((ref[:k, 5] == test[:k, 5]).all())
    return {"n_ref": int(len(ref)), "n_test": int(len(test)), "box_diff_px": box_diff,
            "score_diff": score_diff,
            "passed": len(ref) == len(test) and box_diff < 1.0 and score_diff < 1e-3 and class_match}


def summarize_comparisons(results: list[dict[str, Any]], conf: float) -> dict[str, Any]:
    """Gộp kết quả so sánh của nhiều ảnh thành một dòng báo cáo."""
    return {
        "images": len(results), "conf_threshold": conf,
        "detections_compared": sum(r["n_ref"] for r in results),
        "max_abs_box_diff_px": max(r["box_diff_px"] for r in results),
        "max_abs_score_diff": max(r["score_diff"] for r in results),
        "passed": all(r["passed"] for r in results),
        "failed_images": [i for i, r in enumerate(results) if not r["passed"]],
    }


def verify_numerics(weights: Path, full_onnx: Path, npu_onnx: Path | None,
                    end_nodes: list[dict[str, Any]], test_dir: Path, imgsz: int,
                    count: int, conf: float = 0.25) -> dict[str, Any]:
    """Chạy onnxruntime (CPU) trên ``count`` ảnh test, so với PyTorch / decode NumPy.

    Tiêu chí PASS (ở ngưỡng conf 0.25): cùng số box, cùng lớp, lệch tọa độ < 1 px
    và lệch score < 1e-3 - sai số làm tròn FP32 bình thường nhỏ hơn nhiều bậc.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Cảnh báo] Chưa cài onnxruntime -> bỏ qua kiểm chứng số học.")
        return {"skipped": "onnxruntime not installed"}

    net = load_torch_reference(weights)
    full = ort.InferenceSession(str(full_onnx), providers=["CPUExecutionProvider"])
    npu = (ort.InferenceSession(str(npu_onnx), providers=["CPUExecutionProvider"])
           if npu_onnx is not None else None)
    torch_cmp, npu_cmp, sources = [], [], []
    for x, source in load_test_tensors(test_dir, imgsz, count):
        sources.append(source)
        onnx_out = full.run(None, {full.get_inputs()[0].name: x})[0][0]
        torch_cmp.append(compare_detections(torch_forward(net, x)[0], onnx_out, conf))
        if npu is not None:
            names = [o.name for o in npu.get_outputs()]
            raw = dict(zip(names, npu.run(None, {npu.get_inputs()[0].name: x})))
            decoded = decode_npu_outputs(raw, end_nodes, max_det=onnx_out.shape[0])
            npu_cmp.append(compare_detections(onnx_out, decoded, conf))

    report: dict[str, Any] = {"test_images": sources,
                              "torch_vs_onnx": summarize_comparisons(torch_cmp, conf)}
    if npu_cmp:
        report["onnx_vs_npu_subgraph_plus_cpu_decode"] = summarize_comparisons(npu_cmp, conf)
    return report


# ===========================================================================
# 4. Luồng chính
# ===========================================================================
def file_info(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_MB": round(path.stat().st_size / 2**20, 2), "sha256": digest}


def main() -> int:
    args = parse_args()
    if not args.weights.exists():
        found = sorted((PROJECT_ROOT / "runs" / "detect").glob("*/weights/best*.pt"))
        hint = "\n  ".join(str(p) for p in found) or "(không có)"
        print(f"Lỗi: không tìm thấy {args.weights}\nCác trọng số hiện có:\n  {hint}", file=sys.stderr)
        return 1
    if args.imgsz % 32:
        print(f"Lỗi: --imgsz={args.imgsz} phải là bội số của 32", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    npu_output = args.output.with_name(f"{args.output.stem}_npu.onnx")
    report_path = args.output.with_name(f"{args.output.stem}_onnx_report.json")

    # Bước 1-2: xuất và rút gọn.
    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = export_raw_onnx(args.weights, args.imgsz, args.opset, Path(tmp))
            raw = onnx.load(str(raw_path))
        raw_nodes = len(raw.graph.node)
        model, simplifier = simplify_model(raw, args.simplifier)
        print(f"Rút gọn đồ thị ({simplifier}): {raw_nodes} -> {len(model.graph.node)} node")

        # Bước 3: kiểm tra hợp lệ, lưu ONNX đầy đủ.
        io = validate_model(model)
        onnx.save(model, args.output)
    except (onnx.checker.ValidationError, ValueError, RuntimeError) as exc:
        print(f"Lỗi khi xuất/kiểm tra ONNX: {exc}", file=sys.stderr)
        return 1
    print("onnx.checker.check_model: OK (đồ thị hợp lệ, shape tĩnh, batch=1)")
    print_io(f"ONNX đầy đủ - {args.output.name}", io)

    # Bước 4: phân tích đuôi đồ thị và cắt đồ thị con thuần NPU.
    end_nodes = find_npu_end_nodes(model, args.imgsz)
    analysis = tail_analysis(model, end_nodes)
    npu_io, npu_path = None, None
    if len(end_nodes) != 6:
        print(f"[Cảnh báo] Tìm thấy {len(end_nodes)}/6 Conv cuối của head one-to-one "
              "-> bỏ qua cắt đồ thị NPU (kiểm tra bằng Netron).")
    elif not args.no_npu_subgraph:
        npu_io = extract_npu_subgraph(model, end_nodes, npu_output)
        npu_path = npu_output
        print_io(f"Đồ thị con thuần NPU - {npu_output.name}", npu_io)

    print(f"\n[Phân tích đuôi đồ thị] {analysis['num_nodes']} node; "
          f"{analysis['cpu_tail_nodes']} node sau điểm cắt sẽ chạy trên CPU")
    print(f"  Toán tử khó cho NPU ở đuôi (chuyển sang CPU): {analysis['npu_unfriendly_ops_in_tail']}")
    print(f"  Toán tử khó cho NPU TRƯỚC điểm cắt (cần xem log DFC): "
          f"{analysis['npu_unfriendly_ops_before_cut'] or 'không có'}")
    for n in end_nodes:
        print(f"  end node: {n['node']:<55} {n['kind']:<10} stride {n['stride']:>2} {n['shape']}")

    # Bước 5: kiểm chứng số học.
    verification: dict[str, Any] = {"skipped": "--no-verify"}
    if not args.no_verify:
        verification = verify_numerics(args.weights, args.output, npu_path, end_nodes,
                                       args.test_images, args.imgsz, args.verify_images)
        for name, res in verification.items():
            if isinstance(res, dict):
                print(f"  [Kiểm chứng] {name}: {'PASS' if res['passed'] else 'FAIL'} | "
                      f"{res['images']} ảnh, {res['detections_compared']} box | "
                      f"lệch box max {res['max_abs_box_diff_px']:.2e} px | "
                      f"lệch score max {res['max_abs_score_diff']:.2e}")

    # Bước 6: báo cáo JSON.
    report = {
        "weights": str(args.weights), "imgsz": args.imgsz, "opset": args.opset,
        "simplifier": simplifier, "nms": False, "dynamic": False, "batch": 1,
        "full_onnx": {**file_info(args.output), **io},
        "npu_onnx": {**file_info(npu_path), **npu_io} if npu_path else None,
        "hailo_end_node_names": [n["node"] for n in end_nodes],
        "end_nodes": end_nodes,
        "graph_analysis": analysis,
        "cpu_postprocess": {
            "anchors": "(j + 0.5, i + 0.5) trên lưới H x W của từng stride",
            "box": "x1=(cx-l)*s, y1=(cy-t)*s, x2=(cx+r)*s, y2=(cy+b)*s (reg_max=1, không DFL)",
            "score": "sigmoid(cls_logits)",
            "select": "top max_det cặp (anchor, class) theo score; KHÔNG cần NMS",
            "input_preprocess": "letterbox 114, RGB, /255 (gộp vào NPU qua normalization)",
        },
        "verification": verification,
        "weights_size_MB": round(args.weights.stat().st_size / 2**20, 2),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu: {args.output}\n        {npu_path or '(không cắt đồ thị NPU)'}\n        {report_path}")

    failed = [k for k, v in verification.items() if isinstance(v, dict) and not v.get("passed", True)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
