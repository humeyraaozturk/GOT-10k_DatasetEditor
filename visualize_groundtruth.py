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

def write_groundtruth(gt_path, bboxes):
    with open(gt_path, "w") as f:
        for x, y, w, h in bboxes:
            f.write(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f}\n")

def resize_for_display(frame, max_width=800, max_height=600):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    return resized, scale

mouse_x, mouse_y = -1, -1
def mouse_move(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def select_roi_with_overlay(window_name, image):
    global mouse_x, mouse_y
    clone = image.copy()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_move)

    print("[i] Draw a region to select the object, press ENTER to confirm, C to cancel, A to back, or ESC to exit.")

    # OpenCV ROI seçim aracı ile çizgi overlay
    while True:
        temp = clone.copy()
        if mouse_x >= 0 and mouse_y >= 0:
            # Mouse üzerinde çizgiler
            cv2.line(temp, (mouse_x, 0), (mouse_x, temp.shape[0]), (0, 255, 255), 1)
            cv2.line(temp, (0, mouse_y), (temp.shape[1], mouse_y), (0, 255, 255), 1)
        cv2.imshow(window_name, temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10:  # ENTER → ROI seç
            roi = cv2.selectROI(window_name, clone, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(window_name)
            return roi
        elif key == 27:  # ESC → iptal
            cv2.destroyWindow(window_name)
            return (0, 0, 0, 0)


def main(images_dir):
    gt_path = os.path.join(images_dir, "groundtruth.txt")
    if not os.path.exists(images_dir):
        print("[!] Images folder not found.")
        return
    if not os.path.exists(gt_path):
        print("[!] groundtruth.txt not found.")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp','.webp'))]
    image_files = natsorted(image_files)

    bboxes = read_groundtruth(gt_path)
    total_frames = min(len(image_files), len(bboxes))

    idx = 0
    while 0 <= idx < total_frames:
        img_path = os.path.join(images_dir, image_files[idx])
        frame = cv2.imread(img_path)
        if frame is None:
            idx += 1
            continue

        resized, scale = resize_for_display(frame)

        # Mevcut bbox çizimi
        x, y, w, h = bboxes[idx]
        if w > 0 and h > 0:
            p1 = (int(x * scale), int(y * scale))
            p2 = (int((x + w) * scale), int((y + h) * scale))
            cv2.rectangle(resized, p1, p2, (0, 255, 0), 2)
            cv2.putText(resized, f"Frame {idx} - Label available", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(resized, f"Frame {idx} - Label not available", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


        cv2.imshow("GT Editor", resized)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 13 or key == 10:  # ENTER → Sonraki
            idx += 1
        elif key == ord('a'):  # Geri
            idx = max(0, idx - 1)
        elif key == ord('c'):  # Label sil
            bboxes[idx] = (-1.0, -1.0, -1.0, -1.0)
            write_groundtruth(gt_path, bboxes)
        elif key == ord('w'):  # Yeni seçim
            roi = select_roi_with_overlay("New ROI Selection", resized)
            if roi != (0, 0, 0, 0):
                x_new = roi[0] / scale
                y_new = roi[1] / scale
                w_new = roi[2] / scale
                h_new = roi[3] / scale
                bboxes[idx] = (x_new, y_new, w_new, h_new)
                write_groundtruth(gt_path, bboxes)
                print(f"[✓] Frame {idx}: New label saved.")

    cv2.destroyAllWindows()
    print("[✓] All frames processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Directory of frame images")
    args = parser.parse_args()

    main(args.images_dir)
