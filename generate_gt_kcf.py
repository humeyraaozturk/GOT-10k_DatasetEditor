"""
python generate_gt_kcf.py --video_path ./data/truck.mp4 --output_dir ./got10k_dataset/train/truck_1 --fps 5
"""

import cv2
import os
import argparse

def resize_frame_for_display(frame, max_width=800, max_height=600):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    return resized, scale

def select_roi_with_resize(window_name, frame):
    resized_frame, scale = resize_frame_for_display(frame)
    cv2.imshow(window_name, resized_frame)
    print(f"[i] Nesne seçmek için mouse ile alan seç. Boş bırakmak için ENTER.")

    bbox = cv2.selectROI(window_name, resized_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    if bbox == (0, 0, 0, 0):
        return None

    x, y, w, h = bbox
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)
    return (x, y, w, h)

def create_tracker():
    return cv2.TrackerKCF_create()

def open_clean_file(path):
    if os.path.exists(path):
        os.remove(path)  # Dosya varsa sil
    return open(path, "w")  # Sonra yazma modunda aç


def main(video_path, output_dir, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Video açılamadı.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(video_fps / target_fps)))
    print(f"[i] Video FPS: {video_fps:.2f} → Hedef FPS: {target_fps} → Frame Interval: {frame_interval}")

    os.makedirs(output_dir, exist_ok=True)

    # Dosyaları "w" modunda aç, böylece içerik sıfırlanır
    gt_file = open_clean_file(os.path.join(output_dir, "groundtruth.txt"))
    absence_file = open_clean_file(os.path.join(output_dir, "absence.label"))
    cover_file = open_clean_file(os.path.join(output_dir, "cover.label"))
    cut_file = open_clean_file(os.path.join(output_dir, "cut_by_image.label"))


    frame_idx = 0
    saved_idx = 0

    tracking_active = False
    tracker = None
    bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        if not tracking_active:
            bbox = select_roi_with_resize("Takip Başlat - Nesne Seç", frame)
            if bbox is None:
                print(f"[i] Frame {frame_idx}: Nesne seçilmedi, sonraki frame'e geçiliyor.")
                gt_file.write("0 0 0 0\n")
                absence_file.write("1\n")
                cover_file.write("0\n")
                cut_file.write("0\n")
                cv2.imshow("Takip", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            else:
                tracker = create_tracker()
                tracker.init(frame, bbox)
                x, y, w, h = bbox
                gt_file.write(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f}\n")
                absence_file.write("0\n")
                img_h, img_w = frame.shape[:2]
                cut_file.write("1\n" if (x < 0 or y < 0 or x+w > img_w or y+h > img_h) else "0\n")
                cover_file.write("0\n")

                tracking_active = True
                print(f"[i] Frame {frame_idx}: Takip başlatıldı {bbox}")

        else:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = bbox
                gt_file.write(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f}\n")
                absence_file.write("0\n")
                img_h, img_w = frame.shape[:2]
                cut_file.write("1\n" if (x < 0 or y < 0 or x+w > img_w or y+h > img_h) else "0\n")

                p1 = (int(x), int(y))
                p2 = (int(x + w), int(y + h))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

                cover_file.write("0\n")

            else:
                print(f"[!] Frame {frame_idx}: Takip kayboldu, tekrar seçim yapılacak.")
                tracking_active = False

                # Seçim ekranı aç
                new_bbox = select_roi_with_resize("Takip Kayboldu - Yeni Nesne Seç", frame)
                if new_bbox is None:
                    print(f"[i] Frame {frame_idx}: Yeni nesne seçilmedi, 0 yazılıyor.")
                    gt_file.write("0 0 0 0\n")
                    absence_file.write("1\n")
                    cover_file.write("0\n")
                    cut_file.write("1\n")
                else:
                    tracker = create_tracker()
                    tracker.init(frame, new_bbox)
                    tracking_active = True
                    x, y, w, h = new_bbox
                    gt_file.write(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f}\n")
                    absence_file.write("0\n")
                    cover_file.write("0\n")
                    cut_file.write("0\n")
                    p1 = (int(x), int(y))
                    p2 = (int(x + w), int(y + h))
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    print(f"[i] Frame {frame_idx}: Yeni takip başlatıldı {new_bbox}")

        cv2.imshow("Takip", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        saved_idx += 1

    cap.release()
    gt_file.close()
    absence_file.close()
    cover_file.close()
    cut_file.close()
    cv2.destroyAllWindows()

    print(f"[✓] {saved_idx} frame işlendi ve etiketler kaydedildi → {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Video dosyasının yolu")
    parser.add_argument("--output_dir", required=True, help="Çıktı klasörü (GOT10k formatı)")
    parser.add_argument("--fps", type=int, default=25, help="Çıktı için hedef FPS")
    args = parser.parse_args()

    main(args.video_path, args.output_dir, args.fps)
