import cv2
import glob
import random
import os

# Chọn ngẫu nhiên 5 ảnh từ tập Test
img_paths = random.sample(glob.glob("data/processed/Test/images/*.jpg"), 5)

for img_path in img_paths:
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, cx, cy, bw, bh = map(float, line.strip().split())
                # Chuyển đổi YOLO (chuẩn hóa) sang Pixel
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                
                # Vẽ box màu xanh lá và ghi class_id
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(int(class_id)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Verify", img)
    cv2.waitKey(0) # Nhấn phím bất kỳ để xem ảnh tiếp theo

cv2.destroyAllWindows()