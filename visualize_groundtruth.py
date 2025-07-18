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

drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1
mouse_x, mouse_y = -1, -1
roi_selected = None

def mouse_draw(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, roi_selected, mouse_x, mouse_y
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_x, end_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        roi_selected = (min(start_x, end_x), min(start_y, end_y),
                        abs(end_x - start_x), abs(end_y - start_y))

def select_roi_with_overlay(image):
    global roi_selected, mouse_x, mouse_y
    roi_selected = None
    temp_img = image.copy()
    h, w = temp_img.shape[:2]

    cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Overlay", mouse_draw)

    print("[i] Draw ROI on the scaled image. ENTER = confirm, ESC = cancel")

    while True:
        display = temp_img.copy()

        # Kılavuz çizgiler
        if mouse_x >= 0 and mouse_y >= 0:
            cv2.line(display, (mouse_x, 0), (mouse_x, display.shape[0]), (0, 255, 255), 1)
            cv2.line(display, (0, mouse_y), (display.shape[1], mouse_y), (0, 255, 255), 1)

        # ROI kutusu
        if drawing or roi_selected:
            cv2.rectangle(display, (start_x, start_y), (mouse_x, mouse_y), (0, 255, 0), 2)

        cv2.imshow("Overlay", display)
        key = cv2.waitKey(1) & 0xFF

        if key in [13, 10]:  # ENTER
            cv2.destroyWindow("Overlay")
            return roi_selected if roi_selected else (0, 0, 0, 0)
        elif key == 27:  # ESC
            cv2.destroyWindow("Overlay")
            return (0, 0, 0, 0)

def is_box_out_of_bounds(x, y, w, h, img_w, img_h, margin=5):
    return (x <= margin or y <= margin or
            (x + w) >= (img_w - margin) or
            (y + h) >= (img_h - margin))
    
def update_label_files(images_dir, bboxes):
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp','.webp'))]
    img_files = natsorted(img_files)
    img_path = os.path.join(images_dir, img_files[0])
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    absence_path = os.path.join(images_dir, "absence.label")
    cut_path = os.path.join(images_dir, "cut_by_image.label")
    cover_path = os.path.join(images_dir, "cover.label")

    absence_lines = []
    cut_lines = []
    cover_lines = []

    for bbox in bboxes:
        x, y, w, h = bbox

        # absence.label
        if (x, y, w, h) == (0, 0, 0, 0):
            absence_lines.append("1\n")
            cut_lines.append("0\n")
        else:
            absence_lines.append("0\n")
            # cut_by_image.label
            if w > 0 and h > 0 and is_box_out_of_bounds(x, y, w, h, img_w, img_h):
                cut_lines.append("1\n")
            else:
                cut_lines.append("0\n")

        # cover.label her zaman 0
        cover_lines.append("0\n")

    with open(absence_path, "w") as f:
        f.writelines(absence_lines)
    with open(cut_path, "w") as f:
        f.writelines(cut_lines)
    with open(cover_path, "w") as f:
        f.writelines(cover_lines)

    print("[✓] absence.label, cut_by_image.label, cover.label dosyaları güncellendi.")

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
            bboxes[idx] = (0.0, 0.0, 0.0, 0.0)
            write_groundtruth(gt_path, bboxes)
            update_label_files(images_dir, bboxes)
        elif key == ord('w'):  # Yeni seçim
            roi = select_roi_with_overlay(resized)
            if roi != (0, 0, 0, 0):
                x_new = roi[0] / scale
                y_new = roi[1] / scale
                w_new = roi[2] / scale
                h_new = roi[3] / scale
                bboxes[idx] = (x_new, y_new, w_new, h_new)
                write_groundtruth(gt_path, bboxes)
                update_label_files(images_dir, bboxes)
                print(f"[✓] Frame {idx}: New label saved.")

    cv2.destroyAllWindows()
    print("[✓] All frames processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Directory of frame images")
    args = parser.parse_args()

    main(args.images_dir)
