"""
python convert_to_got10k.py --input_dir ./data --output_dir ./got10k_dataset --fps 5 --split train --yolo_results ./yolo_outputs
"""

import os
import cv2
import argparse

def extract_frames(video_path, output_path, fps):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = f"{saved:08d}.jpg"
            cv2.imwrite(os.path.join(output_path, frame_name), frame)
            saved += 1
        count += 1
    cap.release()
    return saved

def load_yolo_bboxes(yolo_path, frame_count, split):
    if not os.path.exists(yolo_path):
        return ["0 0 0 0"] * frame_count
    with open(yolo_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for i in range(frame_count):
        if i < len(lines):
            values = lines[i].strip().split()
            if len(values) == 5:
                _, x, y, w, h = map(float, values)
            else:
                x, y, w, h = map(float, values[:4])  # fallback
            bboxes.append(f"{x} {y} {w} {h}")
        else:
            if split == "test":
                break
            bboxes.append("0 0 0 0")
    return bboxes

def main(args):
    split_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(split_dir, exist_ok=True)

    for file in os.listdir(args.input_dir):
        if not file.lower().endswith(".mp4"):
            continue

        name = os.path.splitext(file)[0]
        class_name = name.split('_')[0] if '_' in name else name
        out_name = f"{class_name}_1"  # istersen video_id'yi otomatik verebiliriz
        out_path = os.path.join(split_dir, out_name)
        os.makedirs(out_path, exist_ok=True)

        video_path = os.path.join(args.input_dir, file)
        frame_count = extract_frames(video_path, out_path, args.fps)

        # Load YOLO predictions
        yolo_txt = os.path.join(args.yolo_results, f"{name}.txt")
        bboxes = load_yolo_bboxes(yolo_txt, frame_count, args.split)

        gt_path = os.path.join(out_path, "groundtruth.txt")
        with open(gt_path, "w") as f:
            if args.split == "test":
                f.write(bboxes[0] + "\n")
            else:
                for line in bboxes:
                    f.write(line + "\n")

        # Optional empty label files (except test)
        if args.split != "test":
            for label in ["absence.label", "cover.label", "cut_by_image.label"]:
                with open(os.path.join(out_path, label), "w") as f:
                    f.writelines(["0\n"] * frame_count)

        print(f"[✓] {file} → {args.split}/{out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Klasör: Videoların bulunduğu dizin")
    parser.add_argument("--output_dir", required=True, help="GOT-10k ana dizini örn. ./got10k_dataset")
    parser.add_argument("--fps", type=int, default=30, help="Kaç FPS ile frame çıkarılacak")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True, help="Veri bölümü: train, val veya test")
    parser.add_argument("--yolo_results", required=True, help="YOLO çıktılarının bulunduğu klasör")

    args = parser.parse_args()
    main(args)
