import cv2
import os
import argparse
from natsort import natsorted

def resize_frame_for_display(frame, max_width=900, max_height=700):
    """Görüntüyü max boyuta göre ölçekle"""
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))

def read_groundtruth(gt_path):
    bboxes = []
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                x, y, w, h = map(int, map(float, parts))
                bboxes.append((x, y, w, h))
    return bboxes

def crop_frame(frame, bbox):
    x, y, w, h = bbox
    if w > 0 and h > 0:
        return frame[y:y+h, x:x+w]
    return None

def main(images_dir):
    gt_path = os.path.join(images_dir, "groundtruth.txt")
    if not os.path.exists(gt_path):
        print("[!] groundtruth.txt bulunamadı.")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    image_files = natsorted(image_files)

    bboxes = read_groundtruth(gt_path)
    total_frames = len(image_files)

    cover_path = os.path.join(images_dir, "cover.label")
    cover_data = ["\n"] * total_frames

    idx = 0
    cv2.namedWindow("Cover Labeling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cover Labeling", 900, 700)

    while 0 <= idx < total_frames:
        bbox = bboxes[idx]

        # ✅ Eğer groundtruth'ta bbox (0,0,0,0) ise frame atla
        if bbox == (0, 0, 0, 0):
            cover_data[idx] = "0\n"  # Varsayılan olarak 0 (engel yok)
            idx += 1
            continue

        frame = cv2.imread(os.path.join(images_dir, image_files[idx]))
        if frame is None:
            idx += 1
            continue

        cropped = crop_frame(frame, bbox)
        display = cropped if cropped is not None else frame.copy()

        display = resize_frame_for_display(display)
        cv2.putText(display,
                    f"Frame {idx+1}/{total_frames} | [0] No Cover, [1] Covered, [ENTER] Skip, [A] Back",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Cover Labeling", display)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC çık
            break
        elif key == ord('0'):
            cover_data[idx] = "0\n"
            idx += 1
        elif key == ord('1'):
            cover_data[idx] = "1\n"
            idx += 1
        elif key == 13 or key == 10:  # ENTER skip
            if cover_data[idx] == "\n":
                cover_data[idx] = "0\n"
            idx += 1
        elif key == ord('a') and idx > 0:
            idx -= 1

    with open(cover_path, "w") as f:
        f.writelines(cover_data)

    cv2.destroyAllWindows()
    print("[✓] Labelleme tamamlandı:", cover_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    args = parser.parse_args()
    main(args.images_dir)
