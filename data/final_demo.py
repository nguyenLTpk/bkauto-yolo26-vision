import os
import glob
import random
from pathlib import Path
import yaml
import cv2
from ultralytics import YOLO

# 1. Cấu hình đường dẫn
DATASET_YAML = Path("training/configs/dataset.yaml")
WEIGHTS_PATH = Path("runs/detect/yolo26m_baseline/weights/best.pt") # Hoặc đường dẫn file .pt của bạn
TEST_IMG_DIR = Path("data/processed/Test/images")
TEST_LABEL_DIR = Path("data/processed/Test/labels")
OUTPUT_DIR = Path("reports/demo_overlay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Đọc danh sách nhãn từ cấu hình dataset
with open(DATASET_YAML, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
class_names = cfg.get("names", {})
if isinstance(class_names, list):
    class_names = {i: name for i, name in enumerate(class_names)}

# 3. Nạp mô hình
if not WEIGHTS_PATH.exists():
    # Tự động tìm best.pt nếu đường dẫn trên chưa chính xác
    found = list(Path("runs/detect").glob("*/weights/best.pt"))
    if found:
        WEIGHTS_PATH = found[-1]
    else:
        raise FileNotFoundError(f"Không tìm thấy file trọng số {WEIGHTS_PATH}")

print(f"Đang nạp mô hình: {WEIGHTS_PATH}")
model = YOLO(str(WEIGHTS_PATH))

# 4. Lấy danh sách ảnh test (lọc ưu tiên các ảnh có file nhãn không rỗng)
all_images = list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.png"))
valid_samples = []
for p in all_images:
    lbl = TEST_LABEL_DIR / f"{p.stem}.txt"
    if lbl.exists() and lbl.stat().st_size > 0:
        valid_samples.append(p)

# Lấy ngẫu nhiên 5 ảnh có nhãn để demo
samples = random.sample(valid_samples if valid_samples else all_images, min(5, len(all_images)))

# 5. Vẽ nhãn và hiển thị
for idx, img_path in enumerate(samples):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # --- A. VẼ GROUND TRUTH (MÀU XANH LÁ) ---
    label_path = TEST_LABEL_DIR / f"{img_path.stem}.txt"
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = map(float, parts)
                cls_id = int(cls_id)
                name = class_names.get(cls_id, str(cls_id))

                # Đổi tọa độ YOLO sang Pixel
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                # Vẽ nét đứt hoặc nét dày 2px
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"GT: {name}", (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # --- B. VẼ DỰ ĐOÁN TỪ MÔ HÌNH (MÀU ĐỎ) ---
    results = model.predict(source=str(img_path), conf=0.3, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        pred_cls_id = int(box.cls[0])
        pred_name = class_names.get(pred_cls_id, str(pred_cls_id))

        # Vẽ hộp màu đỏ (BGR: 0, 0, 255)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.putText(img, f"Pred: {pred_name} {conf:.2f}", (bx1, min(h - 10, by2 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # --- C. CHÚ THÍCH (LEGEND) Ở GÓC TRÊN CÙNG TRÁI ---
    cv2.rectangle(img, (10, 10), (280, 70), (0, 0, 0), -1) # Nền đen mờ
    cv2.putText(img, "[XANH] Ground Truth (Chuan)", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "[DO]   Model Pred (Du doan)", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

    # Lưu ảnh
    out_file = OUTPUT_DIR / f"overlay_{idx + 1}_{img_path.stem}.jpg"
    cv2.imwrite(str(out_file), img)

    # Hiển thị trực tiếp
    cv2.imshow("Demo: Ground Truth (Xanh) vs Pred (Do) - Nhan phim bat ky de tiep", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"Đã lưu các ảnh kết quả tại: {OUTPUT_DIR.resolve()}")