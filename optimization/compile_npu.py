"""Biên dịch YOLO26m (ONNX, NMS-free) thành HEF cho Hailo-8 bằng Hailo Dataflow Compiler.

Luồng chạy (hàm ``main``), mỗi bước được đo thời gian và ghi log:
    0. Kiểm tra môi trường: ``hailo_sdk_client`` (DFC 3.x - nhánh duy nhất hỗ trợ
       Hailo-8; YOLO26 được Hailo Model Zoo hỗ trợ từ DFC 3.33.1).
    1. Parse (translation): ONNX -> HAR.
         * ``yolo26m_npu.onnx`` (mặc định): đồ thị đã cắt sẵn, parse trực tiếp.
         * ``yolo26m.onnx``: truyền 6 ``hailo_end_node_names`` từ
           ``yolo26m_onnx_report.json`` để DFC tự cắt trước khối decode/TopK.
    2. Model script (.alls): sinh tự động từ ``calibration_data.json``
       (normalization cho ảnh uint8, cấu hình calibration/optimization) + nối
       thêm file ``--allocator-script`` nếu có.
    3. PTQ INT8: ``runner.optimize(calib)`` với 300 ảnh hiệu chuẩn -> quantized HAR.
    4. Compile: ``runner.compile()`` -> ``models/yolo26m.hef`` + compiled HAR.
    5. Phân tích ánh xạ toán tử (NPU vs CPU), đọc HEF (HailoRT), chạy Profiler.
    6. Ghi ``reports/npu_compilation_report.json``.

Môi trường: chỉ chạy được trên Linux x86_64 có Hailo DFC 3.x (Docker "Hailo AI
Software Suite" hoặc virtualenv cài wheel hailo_dataflow_compiler). Nên có GPU
NVIDIA: không có GPU và < 1024 ảnh, DFC tự hạ optimization_level về 0.

Ví dụ:
    python optimization/compile_npu.py
    python optimization/compile_npu.py --onnx models/yolo26m.onnx          # cắt bằng end nodes
    python optimization/compile_npu.py --optimization-level 2 --compiler-opt max
    python optimization/compile_npu.py --dry-run     # kiểm tra artifacts + sinh .alls, không cần DFC
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NPU_ONNX = MODELS_DIR / "yolo26m_npu.onnx"
FULL_ONNX = MODELS_DIR / "yolo26m.onnx"

# Hailo-8 chỉ được DFC nhánh 3.x hỗ trợ (DFC 5.x dành cho Hailo-10H/15H/15L).
# YOLO26 có cấu hình chính thức trong Hailo Model Zoo từ v2.18 (DFC 3.33.1).
MIN_DFC_FOR_YOLO26 = (3, 33)

EXIT_OK, EXIT_FAIL, EXIT_ENV, EXIT_CPU_FALLBACK = 0, 1, 2, 3

LOG = logging.getLogger("compile_npu")

ENV_HELP = """\
Không import được hailo_sdk_client (Hailo Dataflow Compiler).
Kiểm tra:
  1. DFC chỉ chạy trên Linux x86_64 (Ubuntu 22.04/24.04), Python 3.10/3.12.
  2. Docker "Hailo AI Software Suite": chạy ./hailo_ai_sw_suite_docker_run.sh rồi
     thực thi script này BÊN TRONG container.
  3. Hoặc virtualenv riêng:
       python3 -m venv ~/hailo_venv && source ~/hailo_venv/bin/activate
       pip install hailo_dataflow_compiler-3.3x.x-py3-none-linux_x86_64.whl
     (tải wheel từ https://hailo.ai/developer-zone/ - cần tài khoản).
  4. Xác nhận: `pip show hailo-dataflow-compiler`, `hailo --version`,
     python -c "from hailo_sdk_client import ClientRunner".
  5. Hailo-8 cần DFC 3.x (KHÔNG dùng DFC 5.x - chỉ hỗ trợ Hailo-10/15).
Có thể chạy `--dry-run` để kiểm tra artifacts và model script mà không cần DFC."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Biên dịch YOLO26m ONNX -> HEF cho Hailo-8 (Hailo DFC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--onnx", type=Path, default=None,
                        help="ONNX đầu vào (mặc định: yolo26m_npu.onnx, không có thì yolo26m.onnx)")
    parser.add_argument("--calib", type=Path, default=MODELS_DIR / "calibration_data.npy",
                        help="Mảng calibration .npy (N, H, W, 3)")
    parser.add_argument("--calib-json", type=Path, default=MODELS_DIR / "calibration_data.json",
                        help="Manifest của tập calibration")
    parser.add_argument("--report-json", type=Path, default=MODELS_DIR / "yolo26m_onnx_report.json",
                        help="Báo cáo của export_onnx.py (end nodes, opset...)")
    parser.add_argument("--output-hef", type=Path, default=MODELS_DIR / "yolo26m.hef",
                        help="File HEF đầu ra")
    parser.add_argument("--hw-arch", default="hailo8", help="Kiến trúc NPU đích")
    parser.add_argument("--allocator-script", type=Path, default=None,
                        help="File .alls bổ sung, NỐI SAU model script tự sinh "
                             "(resources_param, allocator_param, performance_param...)")
    parser.add_argument("--model-name", default="yolo26m",
                        help="Tên mạng trong HAR/HEF (tiền tố tên layer, vd yolo26m/conv71)")
    parser.add_argument("--build-dir", type=Path, default=MODELS_DIR / "hailo_build",
                        help="Thư mục cache HAR, model script, báo cáo profiler")
    parser.add_argument("--report-out", type=Path, default=REPORTS_DIR / "npu_compilation_report.json",
                        help="Báo cáo tổng kết JSON")
    # --- Lượng tử hóa ---
    parser.add_argument("--optimization-level", type=int, choices=range(5), default=2,
                        help="model_optimization_flavor: 0 nhanh nhất ... 4 chính xác nhất (cần GPU)")
    parser.add_argument("--compression-level", type=int, choices=range(6), default=0,
                        help="Tỉ lệ trọng số 4-bit (0 = tắt, giữ INT8 toàn bộ)")
    parser.add_argument("--post-quant", choices=["finetune", "adaround", "none"], default="finetune",
                        help="Thuật toán sau lượng tử hóa (finetune=QFT, adaround=Hailo Model Zoo)")
    parser.add_argument("--calib-batch", type=int, default=8, help="Batch size khi calibration")
    parser.add_argument("--calib-mode", choices=["array", "dataset"], default="array",
                        help="array: nạp mảng float32 vào RAM (~1.4 GB); "
                             "dataset: tf.data generator từng ảnh (tiết kiệm RAM, thử nghiệm)")
    parser.add_argument("--no-output-16bit", action="store_true",
                        help="Không lượng tử hóa 16-bit cho 6 output layer")
    parser.add_argument("--compiler-opt", choices=["default", "max"], default="default",
                        help="max: compiler_optimization_level=max (FPS cao hơn, biên dịch lâu hơn)")
    # --- Điều khiển ---
    parser.add_argument("--calib-source", type=Path, default=PROJECT_ROOT / "data" / "processed" / "Train" / "images",
                        help="Thư mục ảnh để dựng lại .npy từ manifest nếu thiếu file .npy")
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="Không trả mã lỗi khi phát hiện layer chạy trên CPU")
    parser.add_argument("--skip-profile", action="store_true", help="Bỏ qua Hailo Profiler")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ kiểm tra artifacts + sinh model script, không gọi DFC")
    return parser.parse_args()


# ===========================================================================
# Tiện ích: logging thời gian, hash, JSON
# ===========================================================================
class StepTimer:
    """Ghi log bắt đầu/kết thúc và thời gian (giây) của từng bước biên dịch."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[None]:
        LOG.info("==> %s ...", name)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(time.perf_counter() - start, 2)
            LOG.info("<== %s: %.1f s", name, self.timings[name])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return {"path": str(path), "size_MB": round(path.stat().st_size / 2**20, 2),
            "sha256": sha256_of(path)}


def load_json(path: Path, what: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {what}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def short_name(layer: str) -> str:
    """'yolo26m/output_layer1' -> 'output_layer1' (model script dùng tên không scope)."""
    return layer.split("/")[-1]


# ===========================================================================
# Bước 0: môi trường
# ===========================================================================
def check_environment(hw_arch: str) -> dict[str, Any]:
    """Import DFC (TRƯỚC TensorFlow - yêu cầu của Hailo để dùng được GPU) và kiểm tra phiên bản."""
    try:
        import hailo_sdk_client
    except ImportError as exc:
        raise EnvironmentError(f"{ENV_HELP}\n(Chi tiết: {exc})") from exc
    if not hasattr(hailo_sdk_client, "ClientRunner"):
        raise EnvironmentError(f"{ENV_HELP}\n(hailo_sdk_client không có ClientRunner)")

    version = getattr(hailo_sdk_client, "__version__", "unknown")
    env: dict[str, Any] = {"dfc_version": version, "hw_arch": hw_arch,
                           "python": platform.python_version(), "platform": platform.platform()}
    numbers = tuple(int(x) for x in re.findall(r"\d+", version)[:2])
    if hw_arch.startswith("hailo8") and numbers and numbers[0] >= 5:
        raise EnvironmentError(f"DFC {version} không hỗ trợ {hw_arch}: Hailo-8/8L cần DFC 3.x.")
    if numbers and numbers < MIN_DFC_FOR_YOLO26:
        LOG.warning("DFC %s < 3.33: Hailo Model Zoo chỉ hỗ trợ YOLO26 từ DFC 3.33.1 - "
                    "có thể gặp lỗi parse/optimize.", version)
    with contextlib.suppress(ImportError):
        import hailo_platform
        env["hailort_version"] = getattr(hailo_platform, "__version__", "unknown")
    env["gpu"] = shutil.which("nvidia-smi") is not None
    if not env["gpu"]:
        LOG.warning("Không thấy GPU NVIDIA: với < 1024 ảnh calibration, DFC sẽ tự hạ "
                    "optimization_level về 0 (độ chính xác INT8 có thể giảm).")
    return env


# ===========================================================================
# Artifacts đầu vào
# ===========================================================================
def resolve_onnx(path: Path | None) -> tuple[Path, str]:
    """Chọn ONNX: ưu tiên đồ thị thuần NPU, fallback sang đồ thị đầy đủ."""
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy ONNX: {path}")
        return path, "npu" if path.stem.endswith("_npu") else "full"
    for candidate, kind in ((NPU_ONNX, "npu"), (FULL_ONNX, "full")):
        if candidate.exists():
            return candidate, kind
    raise FileNotFoundError(f"Không có {NPU_ONNX.name} hay {FULL_ONNX.name} trong {MODELS_DIR}. "
                            "Chạy optimization/export_onnx.py trước.")


def input_spec(onnx_report: dict[str, Any]) -> tuple[str, list[int]]:
    """Tên & shape NCHW của input từ báo cáo export (vd 'images', [1, 3, 640, 640])."""
    spec = onnx_report["full_onnx"]["inputs"][0]
    return spec["name"], [int(d) for d in spec["shape"]]


def ensure_calibration(args: argparse.Namespace, manifest: dict[str, Any]) -> np.ndarray:
    """Nạp mảng calibration (memmap, không chiếm RAM) và đối chiếu với manifest.

    Nếu thiếu file .npy (vd chỉ đồng bộ manifest JSON từ server), dựng lại đúng
    300 ảnh đã liệt kê trong manifest bằng cùng hàm tiền xử lý của
    calibrate_ptq.py, rồi so mean/std theo kênh với manifest để chứng minh tái lập.
    """
    expected = tuple(manifest["shape"])
    if not args.calib.exists():
        LOG.warning("Thiếu %s -> dựng lại từ manifest (%d ảnh trong %s).",
                    args.calib, len(manifest["files"]), args.calib_source)
        rebuild_calibration(args.calib, manifest, args.calib_source)

    data = np.load(args.calib, mmap_mode="r")
    if data.shape != expected or str(data.dtype) != manifest["dtype"]:
        raise ValueError(f"Calibration {data.shape}/{data.dtype} không khớp manifest "
                         f"{expected}/{manifest['dtype']}")
    if manifest.get("layout") != "NHWC" or data.shape[-1] != 3:
        raise ValueError("Hailo DFC cần calibration dạng NHWC (N, H, W, 3)")
    return data


def rebuild_calibration(output: Path, manifest: dict[str, Any], source_dir: Path) -> None:
    """Dựng lại calibration .npy từ danh sách file + tham số tiền xử lý trong manifest."""
    from calibrate_ptq import channel_stats, preprocess, to_output_dtype

    n, size = manifest["shape"][0], manifest["shape"][1]
    normalize = manifest["dtype"] == "float32"
    array = np.lib.format.open_memmap(output, mode="w+", dtype=manifest["dtype"],
                                      shape=tuple(manifest["shape"]))
    completed = False
    try:
        for i, name in enumerate(manifest["files"][:n]):
            array[i] = to_output_dtype(preprocess(source_dir / name, size), normalize)
        array.flush()
        completed = True
    finally:
        del array  # đóng memmap (Windows không xóa được file đang map)
        if not completed:
            output.unlink(missing_ok=True)
    stats, ref = channel_stats(output), manifest.get("channel_stats_0_255", {})
    if ref and not np.allclose(stats["mean_rgb"], ref["mean_rgb"], atol=0.01):
        LOG.warning("Mean RGB sau khi dựng lại %s khác manifest %s - dữ liệu nguồn đã thay đổi?",
                    stats["mean_rgb"], ref["mean_rgb"])
    else:
        LOG.info("Dựng lại calibration khớp manifest (mean RGB %s).", stats["mean_rgb"])


def calibration_feed(data: np.ndarray, mode: str) -> Any:
    """Chuẩn bị dữ liệu cho ``runner.optimize``.

    Hailo Model Zoo và Ultralytics đều đưa vào float32 GIỮ NGUYÊN dải 0..255 (KHÔNG
    chia 255) khi model script có lớp ``normalization`` - phép chia do NPU làm.
    - ``array``: chuyển từng khối uint8 -> float32 vào một mảng (300 ảnh ~1.4 GB).
    - ``dataset``: hàm trả về ``tf.data.Dataset`` sinh từng ảnh từ memmap - RAM
      gần như không đổi theo số ảnh (dạng callable giống Hailo Model Zoo; chế độ
      thử nghiệm, nếu DFC báo lỗi định dạng hãy dùng ``array``).
    """
    if mode == "array":
        feed = np.empty(data.shape, dtype=np.float32)
        for start in range(0, len(data), 32):
            feed[start:start + 32] = data[start:start + 32]
        return feed

    import tensorflow as tf  # chỉ import SAU hailo_sdk_client (yêu cầu GPU của Hailo)

    def generator() -> Iterator[tuple[np.ndarray, dict[str, int]]]:
        for i in range(len(data)):
            yield data[i].astype(np.float32), {"index": i}

    signature = (tf.TensorSpec(shape=data.shape[1:], dtype=tf.float32),
                 {"index": tf.TensorSpec(shape=(), dtype=tf.int32)})
    return lambda: tf.data.Dataset.from_generator(generator, output_signature=signature)


# ===========================================================================
# Bước 2: model script
# ===========================================================================
def build_model_script(args: argparse.Namespace, manifest: dict[str, Any],
                       output_layers: list[str], calib_size: int) -> str:
    """Sinh model script (.alls) cho YOLO26m trên Hailo-8.

    Tham khảo cấu hình chính thức ``hailo_model_zoo/cfg/alls/generic/yolo26m.alls``
    và exporter Hailo của Ultralytics cho YOLO26:
      - normalization: ảnh uint8 0..255 -> NPU tự chia 255 (gộp vào layer đầu),
        khớp với calibration_data.json (dtype uint8). Nếu calibration là float32
        [0, 1] thì KHÔNG thêm lớp này (dữ liệu đã được chuẩn hóa trên host).
      - calibset_size: mặc định DFC chỉ dùng 64 ảnh -> khai báo rõ để dùng đủ.
      - optimization_level / post_quantization_optimization: finetune (QFT) như
        Ultralytics hoặc adaround như Model Zoo (chậm, tốn RAM/GPU nhất).
      - Output 16-bit (a16_w16): 6 output là logit lớp & khoảng cách box, dải giá
        trị rộng; lượng tử 16-bit giảm sai số ở bước decode trên CPU (YOLO26 bị
        dịch score ~0.05 khi để INT8 theo tài liệu Ultralytics).
      - KHÔNG có nms_postprocess / sigmoid trên chip: YOLO26 NMS-free, sigmoid +
        decode + top-k chạy trên CPU (xem cpu_postprocess trong onnx report).
    """
    lines = ["# Model script sinh tự động bởi optimization/compile_npu.py"]
    if manifest.get("dtype") == "uint8":
        lines.append(manifest.get("hailo_alls_hint")
                     or "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])")
    lines += [
        f"model_optimization_config(calibration, batch_size={args.calib_batch}, "
        f"calibset_size={calib_size})",
        f"model_optimization_flavor(optimization_level={args.optimization_level}, "
        f"compression_level={args.compression_level})",
    ]
    if args.post_quant == "finetune":
        lines.append(f"post_quantization_optimization(finetune, policy=enabled, dataset_size={calib_size})")
    elif args.post_quant == "adaround":
        lines.append("post_quantization_optimization(adaround, policy=enabled, batch_size=1)")
    if not args.no_output_16bit and output_layers:
        names = ", ".join(short_name(n) for n in output_layers)
        lines.append(f"quantization_param([{names}], precision_mode=a16_w16)")
    if args.compiler_opt == "max":
        lines.append("performance_param(compiler_optimization_level=max)")
    if args.allocator_script:
        if not args.allocator_script.exists():
            raise FileNotFoundError(f"Không tìm thấy allocator script: {args.allocator_script}")
        lines += [f"# --- nối thêm từ {args.allocator_script} ---",
                  args.allocator_script.read_text(encoding="utf-8").strip()]
    return "\n".join(lines) + "\n"


# ===========================================================================
# Bước 1, 3, 4: parse / optimize / compile
# ===========================================================================
def parse_onnx(runner: Any, onnx_path: Path, onnx_kind: str, model_name: str,
               input_name: str, input_shape: list[int], end_nodes: list[str], har: Path) -> None:
    """ONNX -> HN/HAR. Lỗi thường gặp được bổ sung gợi ý xử lý."""
    kwargs: dict[str, Any] = {"start_node_names": [input_name]}
    if onnx_kind == "npu":
        kwargs["net_input_shapes"] = {input_name: input_shape}  # NCHW
    else:
        kwargs["end_node_names"] = end_nodes  # DFC tự cắt trước decode/TopK
    try:
        runner.translate_onnx_model(str(onnx_path), model_name, **kwargs)
    except Exception as exc:
        text = str(exc)
        hints = []
        if re.search(r"unsupported|TopK|GatherElements|Mod|ReduceMax", text, re.I):
            hints.append("Đồ thị còn toán tử hậu xử lý: dùng yolo26m_npu.onnx hoặc kiểm tra end nodes.")
        if re.search(r"opset", text, re.I):
            hints.append("Thử xuất lại ONNX với opset 11 (Ultralytics dùng opset 11 cho Hailo): "
                         "python optimization/export_onnx.py --opset 11")
        raise RuntimeError(f"Parse ONNX thất bại: {text}\n" + "\n".join(hints)) from exc
    runner.save_har(str(har))


def hn_layers(runner: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """(layers, net_params) của HN hiện tại."""
    hn = runner.get_hn()
    hn = json.loads(hn) if isinstance(hn, str) else hn
    return hn.get("layers", {}), hn.get("net_params", {})


def output_layer_names(layers: dict[str, Any]) -> list[str]:
    def order(name: str) -> int:
        digits = re.findall(r"\d+", short_name(name))
        return int(digits[-1]) if digits else 0
    return sorted((n for n, l in layers.items() if l.get("type") == "output_layer"), key=order)


def run_optimize(runner: Any, script: str, feed: Any, har: Path) -> None:
    runner.load_model_script(script)
    runner.optimize(feed)
    runner.save_har(str(har))


def run_compile(runner: Any, hef_path: Path, har: Path) -> None:
    hef_path.parent.mkdir(parents=True, exist_ok=True)
    hef_path.write_bytes(runner.compile())
    runner.save_har(str(har))  # compiled HAR chứa metadata cho profiler


# ===========================================================================
# Bước 5: phân tích ánh xạ toán tử, HEF, profiler
# ===========================================================================
def analyze_mapping(layers: dict[str, Any], net_params: dict[str, Any],
                    end_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Tỉ lệ layer chạy trên NPU và danh sách layer rơi về CPU (nếu có).

    Trên Hailo-8 KHÔNG có cơ chế "CPU fallback" tự động cho toán tử không hỗ trợ:
    toán tử lạ làm bước parse thất bại. Phần chạy trên host chỉ xuất hiện khi
    model script thêm hậu xử lý (vd ``nms_postprocess(..., engine=cpu)``) - khi
    đó HN có layer ``engine == "cpu"`` hoặc ``type == "postprocess"``. Layer
    không khai báo ``engine`` mặc định chạy trên nn_core (NPU).
    """
    io_types = {"input_layer", "output_layer"}
    cpu_layers = sorted(n for n, l in layers.items()
                        if l.get("engine") == "cpu" or l.get("type") == "postprocess")
    compute = [n for n, l in layers.items() if l.get("type") not in io_types]
    npu_compute = [n for n in compute if n not in cpu_layers]
    type_hist: dict[str, int] = {}
    for layer in layers.values():
        type_hist[layer.get("type", "?")] = type_hist.get(layer.get("type", "?"), 0) + 1

    # Ánh xạ output layer -> conv nguồn -> node ONNX gốc (box/cls, stride) cho Giai đoạn 4.
    by_onnx = {n["node"]: n for n in end_nodes}
    outputs = []
    for name in output_layer_names(layers):
        src = (layers[name].get("input") or [None])[0]
        originals = layers.get(src, {}).get("original_names", []) if src else []
        match = next((by_onnx[o] for o in originals if o in by_onnx), None)
        outputs.append({
            "output_layer": name, "source_layer": src, "onnx_node": match["node"] if match else None,
            "kind": match["kind"] if match else None, "stride": match["stride"] if match else None,
            "shape_nhwc": layers[name].get("output_shapes", [None])[0],
        })
    return {
        "total_layers": len(layers),
        "compute_layers": len(compute),
        "npu_compute_layers": len(npu_compute),
        "npu_allocation_rate_pct": round(100.0 * len(npu_compute) / max(1, len(compute)), 2),
        "cpu_fallback_layers": cpu_layers,
        "layer_type_histogram": dict(sorted(type_hist.items(), key=lambda kv: -kv[1])),
        "output_layers_order": net_params.get("output_layers_order"),
        "outputs": outputs,
        "all_end_nodes_mapped": all(o["onnx_node"] for o in outputs) and len(outputs) == len(end_nodes),
    }


def inspect_hef(hef_path: Path) -> dict[str, Any]:
    """Đọc HEF bằng HailoRT (không cần thiết bị) + ``hailortcli parse-hef`` nếu có."""
    info: dict[str, Any] = {}
    try:
        from hailo_platform import HEF
        hef = HEF(str(hef_path))  # HailoRT chỉ nhận str hoặc bytes
        info["inputs"] = [{"name": v.name, "shape": list(v.shape)} for v in hef.get_input_vstream_infos()]
        info["outputs"] = []
        for v in hef.get_output_vstream_infos():
            entry: dict[str, Any] = {"name": v.name, "shape": list(v.shape)}
            with contextlib.suppress(Exception):
                entry["original_names"] = list(hef.get_original_names_from_vstream_name(v.name))
            info["outputs"].append(entry)
        info["network_groups"] = [{"name": g.name, "is_multi_context": bool(g.is_multi_context)}
                                  for g in hef.get_network_groups_infos()]
        with contextlib.suppress(Exception):
            info["bottleneck_fps"] = float(hef.bottleneck_fps())
    except ImportError:
        info["note"] = "hailo_platform (HailoRT) chưa cài - bỏ qua đọc HEF"
    except Exception as exc:  # HEF hỏng / phiên bản HailoRT không khớp
        info["error"] = str(exc)

    if shutil.which("hailortcli"):
        result = subprocess.run(["hailortcli", "parse-hef", str(hef_path)],
                                capture_output=True, text=True, timeout=120)
        match = re.search(r"Number of contexts:\s*(\d+)", result.stdout)
        info["number_of_contexts"] = int(match.group(1)) if match else (
            1 if "Single Context" in result.stdout else None)
    return info


def run_profiler(runner: Any, hef_path: Path, html_path: Path) -> dict[str, Any]:
    """Hailo Profiler: FPS / latency / công suất ƯỚC TÍNH của compiler.

    ``runner.profile()`` trả về dict có ``stats.performance_details`` (theo Hailo
    Model Zoo v2.19). Lưu ý: mô hình nhiều context (YOLO26m trên Hailo-8 ~4
    context) có FPS phụ thuộc hệ thống chủ (băng thông PCIe - Pi 5 chỉ x1), nên
    đây KHÔNG phải số đo thực tế; power thường là "N/A" khi profile tĩnh. Số liệu
    chính thức cho báo cáo phải đo trên RPi5 + Hailo-8 ở Giai đoạn 4.
    """
    export = runner.profile(should_use_logical_layers=True, hef_filename=str(hef_path))
    stats = export.get("stats", {}) if isinstance(export, dict) else {}
    perf = (stats.get("performance_details") or [{}])[0]
    details = stats.get("model_details", {})
    result = {
        "fps_estimated": perf.get("fps"),
        "latency_ms_estimated": perf.get("latency"),
        "power_W_estimated": perf.get("power"),
        "number_of_contexts": perf.get("number_of_contexts"),
        "ops_per_second": perf.get("ops_per_second"),
        "mac_utilization_pct": perf.get("measured_mac_util"),
        "total_ops_per_frame": details.get("total_ops_per_frame"),
        "weights": details.get("weights"),
        "profiling_mode": stats.get("profiler_parameters", {}).get("profiling_mode"),
        "caveat": "Ước tính tĩnh của compiler; FPS thực phụ thuộc PCIe host (Pi 5: x1).",
    }
    try:
        from hailo_sdk_client.tools.profiler.react_report_generator import ReactReportGenerator
        ReactReportGenerator(export, html_path).create_report(should_open_web_browser=False)
        result["html_report"] = str(html_path)
    except Exception as exc:  # báo cáo HTML là phụ, không làm hỏng cả quy trình
        LOG.warning("Không tạo được báo cáo HTML của profiler: %s", exc)
    return result


# ===========================================================================
# Luồng chính
# ===========================================================================
def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    LOG.info("Báo cáo: %s", path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    timer = StepTimer()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "started", "warnings": [],
    }

    # --- Artifacts đầu vào ---
    try:
        onnx_path, onnx_kind = resolve_onnx(args.onnx)
        onnx_report = load_json(args.report_json, "ONNX report")
        manifest = load_json(args.calib_json, "calibration manifest")
        end_nodes = onnx_report.get("end_nodes", [])
        if len(onnx_report.get("hailo_end_node_names", [])) != 6:
            raise ValueError("ONNX report phải có đúng 6 hailo_end_node_names (box+cls x 3 stride)")
        input_name, input_shape = input_spec(onnx_report)
        if manifest["shape"][1:3] != input_shape[2:4]:
            raise ValueError(f"Calibration {manifest['shape']} không khớp input {input_shape}")
        calib = ensure_calibration(args, manifest)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOG.error("Artifacts không hợp lệ: %s", exc)
        return EXIT_FAIL

    calib_size = int(calib.shape[0])
    LOG.info("ONNX: %s (%s) | input %s %s | calibration %s %s",
             onnx_path.name, onnx_kind, input_name, input_shape, calib.shape, calib.dtype)
    report["inputs"] = {
        "onnx": {**file_info(onnx_path), "kind": onnx_kind, "opset": onnx_report.get("opset"),
                 "input_name": input_name, "input_shape_nchw": input_shape},
        "hailo_end_node_names": onnx_report["hailo_end_node_names"],
        "calibration": {**file_info(args.calib), "shape": list(calib.shape), "dtype": str(calib.dtype),
                        "manifest": str(args.calib_json),
                        "channel_stats_0_255": manifest.get("channel_stats_0_255")},
    }
    report["quantization"] = {
        "precision": "INT8 (PTQ)", "optimization_level": args.optimization_level,
        "compression_level": args.compression_level, "post_quant": args.post_quant,
        "calibset_size": calib_size, "calib_batch": args.calib_batch,
        "output_precision": "a16_w16" if not args.no_output_16bit else "a8_w8",
        "normalization_on_npu": manifest.get("dtype") == "uint8",
    }
    alls_path = args.build_dir / f"{args.model_name}.alls"

    if args.dry_run:
        # Không có HN thật -> giả định 6 output layer theo quy ước đặt tên của DFC.
        script = build_model_script(args, manifest, [f"output_layer{i}" for i in range(1, 7)], calib_size)
        alls_path.write_text(script, encoding="utf-8")
        LOG.info("[dry-run] Model script:\n%s", script)
        report.update(status="dry_run", model_script={"path": str(alls_path), "content": script.splitlines()})
        write_report(args.report_out, report)
        return EXIT_OK

    # --- Bước 0: môi trường ---
    try:
        report["environment"] = check_environment(args.hw_arch)
    except EnvironmentError as exc:
        LOG.error("%s", exc)
        report["status"] = "environment_error"
        write_report(args.report_out, report)
        return EXIT_ENV
    from hailo_sdk_client import ClientRunner

    parsed_har = args.build_dir / f"{args.model_name}_parsed.har"
    quant_har = args.build_dir / f"{args.model_name}_quantized.har"
    compiled_har = args.build_dir / f"{args.model_name}_compiled.har"
    runner = None
    exit_code = EXIT_FAIL
    try:
        runner = ClientRunner(hw_arch=args.hw_arch)

        with timer.step("1. Parse ONNX -> HAR"):
            parse_onnx(runner, onnx_path, onnx_kind, args.model_name, input_name, input_shape,
                       onnx_report["hailo_end_node_names"], parsed_har)
            layers, _ = hn_layers(runner)
            outputs = output_layer_names(layers)
            LOG.info("HN: %d layer, %d output layer", len(layers), len(outputs))
            if len(outputs) != 6:
                raise RuntimeError(f"Mong đợi 6 output layer sau parse, nhận {len(outputs)}: {outputs}")

        with timer.step("2. Model script"):
            script = build_model_script(args, manifest, outputs, calib_size)
            alls_path.write_text(script, encoding="utf-8")
            LOG.info("Model script (%s):\n%s", alls_path, script)
            report["model_script"] = {"path": str(alls_path), "content": script.splitlines()}

        with timer.step("3. PTQ INT8 optimize"):
            run_optimize(runner, script, calibration_feed(calib, args.calib_mode), quant_har)

        with timer.step("4. Compile -> HEF"):
            run_compile(runner, args.output_hef, compiled_har)

        with timer.step("5. Phân tích ánh xạ & HEF"):
            layers, net_params = hn_layers(runner)
            mapping = analyze_mapping(layers, net_params, end_nodes)
            report["mapping"] = mapping
            report["hef_info"] = inspect_hef(args.output_hef)
            LOG.info("NPU allocation: %.2f%% (%d/%d layer tính toán)",
                     mapping["npu_allocation_rate_pct"], mapping["npu_compute_layers"],
                     mapping["compute_layers"])
            if mapping["cpu_fallback_layers"]:
                msg = f"CPU FALLBACK: {len(mapping['cpu_fallback_layers'])} layer chạy trên host: " \
                      f"{mapping['cpu_fallback_layers']}"
                LOG.warning(msg)
                report["warnings"].append(msg)
            else:
                LOG.info("Không có layer CPU fallback: đồ thị 100% chạy trên Hailo-8.")
            if not mapping["all_end_nodes_mapped"]:
                report["warnings"].append("Không ánh xạ đủ 6 output về end node ONNX - kiểm tra HN.")

        if not args.skip_profile:
            with timer.step("6. Hailo Profiler"):
                try:
                    report["profiler"] = run_profiler(
                        runner, args.output_hef, args.build_dir / f"{args.model_name}_profiler.html")
                    LOG.info("Profiler (ước tính): %s FPS, %s ms, %s context",
                             report["profiler"]["fps_estimated"],
                             report["profiler"]["latency_ms_estimated"],
                             report["profiler"]["number_of_contexts"])
                except Exception as exc:
                    LOG.warning("Profiler thất bại (không ảnh hưởng HEF): %s", exc)
                    report["profiler"] = {"error": str(exc)}

        report["status"] = "success"
        exit_code = (EXIT_CPU_FALLBACK if report["mapping"]["cpu_fallback_layers"]
                     and not args.allow_cpu_fallback else EXIT_OK)
    except Exception as exc:
        LOG.exception("Biên dịch thất bại: %s", exc)
        report.update(status="failed", error=str(exc))
    finally:
        # Giải phóng runner (giữ đồ thị, trọng số, session TensorFlow/GPU).
        del runner
        if "tensorflow" in sys.modules:
            with contextlib.suppress(Exception):
                sys.modules["tensorflow"].keras.backend.clear_session()
        gc.collect()

        report["timings_s"] = timer.timings
        report["outputs"] = {"hef": file_info(args.output_hef), "parsed_har": file_info(parsed_har),
                             "quantized_har": file_info(quant_har), "compiled_har": file_info(compiled_har)}
        hef = report["outputs"]["hef"]
        report["footprint_MB"] = {"pt": onnx_report.get("weights_size_MB"),
                                  "onnx": report["inputs"]["onnx"]["size_MB"],
                                  "hef": hef["size_MB"] if hef else None}
        write_report(args.report_out, report)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
