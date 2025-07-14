"""
python visualize_groundtruth.py --images_dir ./got10k_dataset/train/truck_1/ --gt_path ./got10k_dataset/train/truck_1/groundtruth.txt
"""

import cv2
import os
import argparse
from natsort import natsorted

def read_groundtruth(gt_path):
    bboxes = []
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                x, y, w, h = map(float, parts)
                bboxes.append((x, y, w, h))
    return bboxes

def main(images_dir, gt_path):
    if not os.path.exists(images_dir):
        print("[!] Görseller klasörü bulunamadı.")
        return
    if not os.path.exists(gt_path):
        print("[!] groundtruth.txt bulunamadı.")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))]
    image_files = natsorted(image_files)

    bboxes = read_groundtruth(gt_path)
    total_frames = min(len(image_files), len(bboxes))
    print(f"[i] {total_frames} frame gösterilecek.")

    idx = 0

    while idx < total_frames:
        img_path = os.path.join(images_dir, image_files[idx])
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[!] Görsel açılamadı: {img_path}")
            idx += 1
            continue

        x, y, w, h = bboxes[idx]
        if w > 0 and h > 0:
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame {idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"Frame {idx} - No Target", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("GT Görüntüleme", frame)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC çıkış
            break
        elif key == 13 or key == 10:  # ENTER veya RETURN bir sonraki frame
            idx += 1

    cv2.destroyAllWindows()
    print("[✓] Tüm frame'ler gösterildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Frame görüntülerinin olduğu klasör")
    parser.add_argument("--gt_path", required=True, help="groundtruth.txt dosyasının yolu")
    args = parser.parse_args()

    main(args.images_dir, args.gt_path)
