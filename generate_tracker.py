import cv2
import os
import argparse
from ultralytics import YOLO
import numpy as np

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

def open_clean_file(path):
    if os.path.exists(path):
        os.remove(path)
    return open(path, "w")

def iou(box1, box2):
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
    return x <= 5 or y <= 5 or (x + w) >= img_w-5 or (y + h) >= img_h-5

def main(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Video açılamadı.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(video_fps / video_fps)))

    os.makedirs(output_dir, exist_ok=True)

    gt_file = open_clean_file(os.path.join(output_dir, "groundtruth.txt"))
    absence_file = open_clean_file(os.path.join(output_dir, "absence.label"))
    cover_file = open_clean_file(os.path.join(output_dir, "cover.label"))
    cut_file = open_clean_file(os.path.join(output_dir, "cut_by_image.label"))

    model = YOLO("./model_weights/best.pt")
    frame_idx = 0
    target_class = None
    target_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        results = model(frame)[0]
        detections = [(int(b.cls), b.xyxy[0].cpu().numpy()) for b in results.boxes]

        if target_box is None:
            for _, xyxy in detections:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            roi = select_roi_with_resize("Hedef Nesneyi Seç", frame)
            if roi is None:
                gt_file.write("-1 -1 -1 -1\n")
                absence_file.write("1\n")
                cover_file.write("0\n")
                cut_file.write("0\n")
                continue
            best_iou, best_det = 0, None
            for cls, xyxy in detections:
                box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])]
                if iou(roi, box) > best_iou:
                    best_iou = iou(roi, box)
                    best_det = (cls, box)
            if best_det:
                target_class, target_box = best_det

        matched = None
        max_iou = 0
        for cls, xyxy in detections:
            if cls != target_class:
                continue
            box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])]
            i = iou(target_box, box)
            if i > max_iou:
                max_iou = i
                matched = box

        img_h, img_w = frame.shape[:2]
        if matched:
            target_box = matched
            x, y, w, h = matched
            gt_file.write(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f}\n")
            absence_file.write("0\n")
            cut_file.write("1\n" if is_box_out_of_bounds(x, y, w, h, img_w, img_h) else "0\n")
            cover_file.write("0\n")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            gt_file.write("-1 -1 -1 -1\n")
            absence_file.write("1\n")
            cover_file.write("0\n")
            cut_file.write("1\n")

        cv2.imshow("YOLO Takip", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    gt_file.close()
    absence_file.close()
    cover_file.close()
    cut_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args.video_path, args.output_dir)
