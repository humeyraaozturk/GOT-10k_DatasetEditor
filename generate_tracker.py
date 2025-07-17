import cv2
import os
import argparse
from ultralytics import YOLO
import numpy as np


def resize_frame_for_display(frame, max_width=900, max_height=700):
    """Frame ekrana sığmazsa ölçekle."""
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    return resized, scale

def select_roi(window_name, frame):
    """ROI seçimi için pencere aç."""
    resized_frame, scale = resize_frame_for_display(frame)
    bbox = cv2.selectROI(window_name, resized_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    if bbox == (0, 0, 0, 0):
        return None

    x, y, w, h = bbox
    return (int(x / scale), int(y / scale), int(w / scale), int(h / scale))

def open_clean_file(path):
    """Dosyayı sıfırlayarak aç."""
    if os.path.exists(path):
        os.remove(path)
    return open(path, "w")

def iou(box1, box2):
    """İki bbox arasındaki IoU değeri."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[0]+box1[2], box2[0]+box2[2])
    yi2 = min(box1[1]+box1[3], box2[1]+box2[3])
    inter_area = max((xi2 - xi1), 0) * max((yi2 - yi1), 0)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def is_box_out_of_bounds(x, y, w, h, img_w, img_h):
    """BBox görüntü sınırını aşıyor mu?"""
    return x <= 5 or y <= 5 or (x + w) >= img_w - 5 or (y + h) >= img_h - 5

def main(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Video açılamadı.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Dosyaları aç
    gt_file = open_clean_file(os.path.join(output_dir, "groundtruth.txt"))
    absence_file = open_clean_file(os.path.join(output_dir, "absence.label"))
    cover_file = open_clean_file(os.path.join(output_dir, "cover.label"))
    cut_file = open_clean_file(os.path.join(output_dir, "cut_by_image.label"))

    # YOLO modeli
    model = YOLO("./model_weights/best.pt")

    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"[INFO] Video FPS: {video_fps}")

    target_class = None
    target_box = None
    frame_idx = 0
    tracking_active = False

    print("[INFO] Başlangıç: Nesne seçmek için 'W' tuşuna bas, ENTER ile frame geç, ESC ile çık.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video bitti.")
            break

        frame_idx += 1
        img_h, img_w = frame.shape[:2]

        if not tracking_active:
            # Tahmin al
            results = model(frame)[0]
            detections = [(int(b.cls), b.xyxy[0].cpu().numpy()) for b in results.boxes]

            # YOLO kutularını çiz
            for _, xyxy in detections:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            resized, _ = resize_frame_for_display(frame)
            cv2.imshow("Seçim Bekleniyor", resized)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC çıkış
                break
            elif key == 13 or key == 10:  # ENTER: Bu kareyi atla
                gt_file.write("0 0 0 0\n")
                absence_file.write("1\n")
                cover_file.write("0\n")
                cut_file.write("0\n")
                continue
            elif key == ord('w'):  # ROI seçimi
                roi = select_roi("Nesne Seç", frame)
                if roi is None:
                    print("[!] ROI seçilmedi.")
                    gt_file.write("0 0 0 0\n")
                    absence_file.write("1\n")
                    cover_file.write("0\n")
                    cut_file.write("0\n")
                    continue

                # ROI'ye en yakın kutuyu bul
                best_iou, best_det = 0, None
                for cls, xyxy in detections:
                    box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])]
                    score = iou(roi, box)
                    if score > best_iou:
                        best_iou = score
                        best_det = (cls, box)

                if best_det:
                    target_class, target_box = best_det
                    tracking_active = True
                    print(f"[INFO] Takip Başladı: Sınıf={target_class}, Box={target_box}")

                    x, y, w, h = target_box
                    gt_file.write(f"{x} {y} {w} {h}\n")
                    absence_file.write("0\n")
                    cut_file.write("1\n" if is_box_out_of_bounds(x, y, w, h, img_w, img_h) else "0\n")
                    cover_file.write("0\n")
                    
                    continue 
                else:
                    print("[!] Uygun nesne bulunamadı.")
                    gt_file.write("0 0 0 0\n")
                    absence_file.write("1\n")
                    cover_file.write("0\n")
                    cut_file.write("0\n")
                    continue

        else:
            # Takip modu
            results = model(frame)[0]
            detections = [(int(b.cls), b.xyxy[0].cpu().numpy()) for b in results.boxes]

            matched = None
            max_iou = 0
            for cls, xyxy in detections:
                if cls != target_class:
                    continue
                box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])]
                score = iou(target_box, box)
                if score > max_iou:
                    max_iou = score
                    matched = box

            if matched:
                target_box = matched
                x, y, w, h = matched
                gt_file.write(f"{x} {y} {w} {h}\n")
                absence_file.write("0\n")
                cut_file.write("1\n" if is_box_out_of_bounds(x, y, w, h, img_w, img_h) else "0\n")
                cover_file.write("0\n")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            else:
                gt_file.write("0 0 0 0\n")
                absence_file.write("1\n")
                cover_file.write("0\n")
                cut_file.write("0\n")

            cv2.imshow("Takip Ediliyor", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    gt_file.close()
    absence_file.close()
    cover_file.close()
    cut_file.close()
    cv2.destroyAllWindows()
    print(f"[✓] İşlem tamamlandı. Dosyalar: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Video dosyasının yolu")
    parser.add_argument("--output_dir", required=True, help="Çıktı klasörü")
    args = parser.parse_args()
    main(args.video_path, args.output_dir)
