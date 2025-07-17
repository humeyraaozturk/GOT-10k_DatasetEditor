import cv2
import os
import argparse
from ultralytics import YOLO
import numpy as np


def resize_frame_for_display(frame, max_width=900, max_height=700):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    return resized, scale

def select_roi(window_name, frame):
    resized_frame, scale = resize_frame_for_display(frame)
    bbox = cv2.selectROI(window_name, resized_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    if bbox == (0, 0, 0, 0):
        return None

    x, y, w, h = bbox
    return (int(x / scale), int(y / scale), int(w / scale), int(h / scale))

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
    if inter_area <= 0:
        return 0.0
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def is_box_out_of_bounds(x, y, w, h, img_w, img_h):
    return x <= 5 or y <= 5 or (x + w) >= img_w - 5 or (y + h) >= img_h - 5

def main(video_path, output_dir, model_path="./model_weights/best.pt"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Video could not be opened.")
        return

    os.makedirs(output_dir, exist_ok=True)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total frames: {num_frames}")

    gt = []
    absence = []
    cover = []
    cut = []

    model = YOLO(model_path)

    target_class = None
    target_box = None
    tracking_active = False

    current_idx = 0
    print("[INFO] Start: W=Select, ENTER=Skip, A=Back, ESC=Exit")

    # ---------- selection phase ----------
    while not tracking_active and current_idx < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ok, frame = cap.read()
        if not ok:
            print("[!] Frame read failed at index", current_idx)
            break

        img_h, img_w = frame.shape[:2]

        # YOLO detect for visual aid
        result = model(frame)[0]
        detections = [(int(b.cls), b.xyxy[0].cpu().numpy()) for b in result.boxes]

        vis = frame.copy()
        for _, xyxy in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        disp, _ = resize_frame_for_display(vis)
        cv2.putText(disp, f"Frame {current_idx+1}/{num_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50,255,50), 2)
        cv2.putText(disp, "W=Select  ENTER=Skip  A=Back  ESC=Exit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Selection Waiting", disp)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            print("[INFO] Exit during selection.")
            break

        elif key in (13, 10):  # ENTER skip
            gt.append((0,0,0,0))
            absence.append(1)
            cover.append(0)
            cut.append(0)
            current_idx += 1
            continue

        elif key == ord('a'):  # BACK
            if current_idx == 0:
                print("[i] Already at first frame.")
                continue
            # pop last labels (previous frame)
            if gt:
                gt.pop()
                absence.pop()
                cover.pop()
                cut.pop()
            current_idx -= 1
            print(f"[INFO] Back to frame {current_idx}")
            continue

        elif key == ord('w'):  # selection
            roi = select_roi("Object Select", frame)
            if roi is None:
                print("[!] ROI not selected. Marking skipped.")
                gt.append((0,0,0,0))
                absence.append(1)
                cover.append(0)
                cut.append(0)
                current_idx += 1
                continue

            # match ROI to YOLO det
            best_i = 0.0
            best_det = None
            for cls, xyxy in detections:
                box = [int(xyxy[0]), int(xyxy[1]),
                       int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])]
                sc = iou(roi, box)
                if sc > best_i:
                    best_i = sc
                    best_det = (cls, box)

            if best_det is None:
                print("[!] No matching det. Mark skip.")
                gt.append((0,0,0,0))
                absence.append(1)
                cover.append(0)
                cut.append(0)
                current_idx += 1
                continue

            # lock target
            target_class, target_box = best_det
            tracking_active = True
            print(f"[INFO] Tracking started at frame {current_idx}  class={target_class} box={target_box}")

            # write this frame label
            x,y,w,h = target_box
            gt.append((x,y,w,h))
            absence.append(0)
            cover.append(0)
            cut.append(1 if is_box_out_of_bounds(x,y,w,h,img_w,img_h) else 0)

            current_idx += 1  # advance to next frame; we'll switch to tracking loop
            break

    # if never entered tracking, finish & write what we have
    if not tracking_active:
        print("[INFO] No tracking started; writing what was labeled and exiting.")
        _write_labels(output_dir, gt, absence, cover, cut)
        cap.release()
        cv2.destroyAllWindows()
        return

    # ---------- tracking phase ----------
    while current_idx < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ok, frame = cap.read()
        if not ok:
            break

        img_h, img_w = frame.shape[:2]
        result = model(frame)[0]
        detections = [(int(b.cls), b.xyxy[0].cpu().numpy()) for b in result.boxes]

        matched = None
        best_i = 0.0
        for cls, xyxy in detections:
            if cls != target_class:
                continue
            box = [int(xyxy[0]), int(xyxy[1]),
                   int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])]
            sc = iou(target_box, box)
            if sc > best_i:
                best_i = sc
                matched = box

        if matched:
            target_box = matched
            x,y,w,h = matched
            gt.append((x,y,w,h))
            absence.append(0)
            cover.append(0)
            cut.append(1 if is_box_out_of_bounds(x,y,w,h,img_w,img_h) else 0)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        else:
            gt.append((0,0,0,0))
            absence.append(1)
            cover.append(0)
            cut.append(0)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC stop tracking early
            print("[INFO] Stopped early by user.")
            break

        current_idx += 1

    # pad any missing frames (if user ESC early)
    while current_idx < num_frames:
        gt.append((0,0,0,0))
        absence.append(1)
        cover.append(0)
        cut.append(0)
        current_idx += 1

    _write_labels(output_dir, gt, absence, cover, cut)
    cap.release()
    cv2.destroyAllWindows()
    print(f"[✓] Done. Labels written to: {output_dir}")

# ---------- label writer ----------
def _write_labels(output_dir, gt, absence, cover, cut):
    # ensure equal length
    n = min(len(gt), len(absence), len(cover), len(cut))
    gt = gt[:n]; absence = absence[:n]; cover = cover[:n]; cut = cut[:n]

    with open(os.path.join(output_dir,"groundtruth.txt"),"w") as f:
        for x,y,w,h in gt:
            f.write(f"{x} {y} {w} {h}\n")

    with open(os.path.join(output_dir,"absence.label"),"w") as f:
        for v in absence:
            f.write(f"{v}\n")

    with open(os.path.join(output_dir,"cover.label"),"w") as f:
        for v in cover:
            f.write(f"{v}\n")

    with open(os.path.join(output_dir,"cut_by_image.label"),"w") as f:
        for v in cut:
            f.write(f"{v}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Video file path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    main(args.video_path, args.output_dir)
