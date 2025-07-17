import os
import cv2
import argparse

def draw_cross(img, x, y):
    cv2.line(img, (x, 0), (x, img.shape[0]), (0, 255, 0), 1)
    cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)

def resize_for_display(frame, max_width=900, max_height=600):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)  # Daha büyükse küçült, küçükse aynı kal
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    return resized, scale

def interactive_selection(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Video could not be opened:", video_path)
        return None, 0

    frame_idx = 0
    roi = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Video end reached, no selection made.")
            break

        frame_name = f"{frame_idx:08d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)

        scaled_frame, scale = resize_for_display(frame)
        crosshair_pos = [scaled_frame.shape[1]//2, scaled_frame.shape[0]//2]
        selecting = False
        start_x, start_y = -1, -1
        temp = scaled_frame.copy()

        def mouse_callback(event, x, y, flags, param):
            nonlocal selecting, start_x, start_y, roi, temp, crosshair_pos
            crosshair_pos[0], crosshair_pos[1] = x, y

            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                start_x, start_y = x, y
                roi = None
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                x1, y1 = min(start_x, x), min(start_y, y)
                w, h = abs(start_x - x), abs(start_y - y)
                roi = (x1, y1, w, h)
            elif event == cv2.EVENT_LBUTTONUP:
                selecting = False
                x1, y1 = min(start_x, x), min(start_y, y)
                w, h = abs(start_x - x), abs(start_y - y)
                roi = (x1, y1, w, h)

        cv2.namedWindow("ROI Selection (Enter: Next, C: Clear, ESC: Exit)")
        cv2.setMouseCallback("ROI Selection (Enter: Next, C: Clear, ESC: Exit)", mouse_callback)

        while True:
            display_frame = temp.copy()
            # Crosshair çiz
            cv2.line(display_frame, (crosshair_pos[0], 0), (crosshair_pos[0], display_frame.shape[0]), (0, 255, 0), 1)
            cv2.line(display_frame, (0, crosshair_pos[1]), (display_frame.shape[1], crosshair_pos[1]), (0, 255, 0), 1)

            # ROI çiz
            if roi and roi[2] > 0 and roi[3] > 0:
                x, y, w, h = roi
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("ROI Selection (Enter: Next, C: Clear, ESC: Exit)", display_frame)

            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # ENTER
                if roi and roi[2] > 0 and roi[3] > 0:
                    real_x = int(roi[0] / scale)
                    real_y = int(roi[1] / scale)
                    real_w = int(roi[2] / scale)
                    real_h = int(roi[3] / scale)
                    cap.release()
                    cv2.destroyAllWindows()
                    return (real_x, real_y, real_w, real_h), frame_idx
                else:
                    print(f"[i] Frame {frame_idx}: Selection not made, moving to next frame.")
                    break
            elif key == ord('c'):
                roi = None
                print("[i] Selection cleared.")
            elif key == 27:  # ESC
                print("[i] Operation canceled.")
                cap.release()
                cv2.destroyAllWindows()
                return None, -1

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return None, -1

def extract_remaining_frames(video_path, start_frame_idx, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx + 1)
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"{start_frame_idx + 1 + saved:08d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        saved += 1

    cap.release()
    return saved

def extract_all_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"{frame_idx:08d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        frame_idx += 1
    cap.release()
    return frame_idx

def main(args):
    split_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(split_dir, exist_ok=True)

    for file in os.listdir(args.input_dir):
        if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        name = os.path.splitext(file)[0]
        class_name = name.split('_')[0] if '_' in name else name
        out_name = f"{class_name}_1"
        out_path = os.path.join(split_dir, out_name)
        os.makedirs(out_path, exist_ok=True)

        video_path = os.path.join(args.input_dir, file)

        if args.split in ["train", "val"]:
            total_frames = extract_all_frames(video_path, out_path)
            for label_file in ["absence.label", "cover.label", "cut_by_image.label", "groundtruth.txt"]:
                open(os.path.join(out_path, label_file), "w").close()  # boş dosyalar
            print(f"[✓] {file} → {args.split}/{out_name} ({total_frames} frame)")
        else:  # test
            print(f"Selection mode for [i] {file} ...")
            roi, selected_frame = interactive_selection(video_path, out_path)
            if roi is not None and selected_frame >= 0:
                x, y, w, h = roi
                # Negatif değerleri düzelt
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
                ret, frame = cap.read()
                cap.release()
                frame_h, frame_w = frame.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)
                with open(os.path.join(out_path, "groundtruth.txt"), "w") as f:
                    f.write(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f}\n")
                remaining = extract_remaining_frames(video_path, selected_frame, out_path)
                print(f"[✓] ROI saved ({x},{y},{w},{h}) → {out_name} | Total frames: {selected_frame + 1 + remaining}")
            else:
                print("[!] ROI not selected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Video directory")
    parser.add_argument("--output_dir", required=True, help="GOT10k main directory")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = parser.parse_args()

    main(args)
