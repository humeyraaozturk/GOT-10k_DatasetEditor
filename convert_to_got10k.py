import os
import cv2
import argparse
from ultralytics import YOLO

def calc_step(real_fps, desired_fps):
    if desired_fps is None or desired_fps <= 0 or desired_fps > real_fps:
        desired_fps = real_fps
    return max(1, int(round(real_fps / desired_fps))), desired_fps

def resize_for_display(frame, max_width=900, max_height=600):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h)), scale

def draw_cross(img, x, y):
    cv2.line(img, (x, 0), (x, img.shape[0]), (0, 255, 0), 1)
    cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)

def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[0] + box1[2], box2[0] + box2[2])
    yi2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter_w = max(xi2 - xi1, 0)
    inter_h = max(yi2 - yi1, 0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    a1 = box1[2] * box1[3]
    a2 = box2[2] * box2[3]
    u = a1 + a2 - inter_area
    return inter_area / u if u > 0 else 0.0

def is_box_out_of_bounds(x, y, w, h, img_w, img_h, margin=5):
    return (x <= margin or y <= margin or
            (x + w) >= (img_w - margin) or
            (y + h) >= (img_h - margin))

def write_lines(path, lines):
    with open(path, "w") as f:
        f.writelines(lines)

def write_labels_train_val(output_dir, gt, absence, cover, cut):
    # ensure aligned lengths
    n = min(len(gt), len(absence), len(cover), len(cut))
    gt, absence, cover, cut = gt[:n], absence[:n], cover[:n], cut[:n]

    write_lines(os.path.join(output_dir, "groundtruth.txt"),
                [f"{x:.4f},{y:.4f},{w:.4f},{h:.4f}\n" for (x, y, w, h) in gt])
    write_lines(os.path.join(output_dir, "absence.label"),
                [f"{v}\n" for v in absence])
    write_lines(os.path.join(output_dir, "cover.label"),
                [f"{v}\n" for v in cover])
    write_lines(os.path.join(output_dir, "cut_by_image.label"),
                [f"{v}\n" for v in cut])

def write_labels_test(output_dir, x, y, w, h):
    # GOT-10k test: only first frame GT (one line)
    with open(os.path.join(output_dir, "groundtruth.txt"), "w") as f:
        f.write(f"{x:.4f},{y:.4f},{w:.4f},{h:.4f}\n")

def interactive_roi_select_on_scaled(frame_scaled, scale, yolo_results=None,
                                     win_name="ROI Select",
                                     crosshair=True):
    """
    frame_scaled: display için küçültülmüş BGR görüntü.
    scale: original_to_scaled = scale. (scaled = orig * scale)
           -> orig = int(scaled / scale)
    yolo_results: iterable of (x1,y1,x2,y2) in ORIGINAL coords.
    return: (x,y,w,h) in ORIGINAL coords, or None.
    """
    disp = frame_scaled.copy()

    # YOLO kutuları (orijinal → scaled)
    if yolo_results:
        for (x1, y1, x2, y2) in yolo_results:
            sx1 = int(x1 * scale)
            sy1 = int(y1 * scale)
            sx2 = int(x2 * scale)
            sy2 = int(y2 * scale)
            cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

    cross_pos = [disp.shape[1] // 2, disp.shape[0] // 2]
    selecting = False
    start_x, start_y = -1, -1
    roi_scaled = None

    def _cb(event, x, y, flags, param):
        nonlocal selecting, start_x, start_y, roi_scaled, cross_pos
        cross_pos[0], cross_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_x, start_y = x, y
            roi_scaled = None
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            x1, y1 = min(start_x, x), min(start_y, y)
            w, h = abs(start_x - x), abs(start_y - y)
            roi_scaled = (x1, y1, w, h)
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            x1, y1 = min(start_x, x), min(start_y, y)
            w, h = abs(start_x - x), abs(start_y - y)
            roi_scaled = (x1, y1, w, h)

    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)  # scaled görüntü boyutunda açılır
    cv2.setMouseCallback(win_name, _cb)

    while True:
        show = disp.copy()
        if crosshair:
            draw_cross(show, cross_pos[0], cross_pos[1])
        if roi_scaled and roi_scaled[2] > 0 and roi_scaled[3] > 0:
            x, y, w, h = roi_scaled
            cv2.rectangle(show, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(show, "[ENTER]=Onay  [C]=Temizle  [ESC]=Iptal",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(win_name, show)
        k = cv2.waitKey(20) & 0xFF
        if k == 13:      # Enter
            break
        elif k == ord('c'):
            roi_scaled = None
        elif k == 27:    # ESC
            roi_scaled = None
            break

    cv2.destroyWindow(win_name)

    if not roi_scaled or roi_scaled[2] <= 0 or roi_scaled[3] <= 0:
        return None

    sx, sy, sw, sh = roi_scaled
    # scaled → original
    ox = int(sx / scale)
    oy = int(sy / scale)
    ow = int(sw / scale)
    oh = int(sh / scale)
    return (ox, oy, ow, oh)

# ------------------------------------------------------------
# TRAIN / VAL: interactive tracking + labeling + frame save
# ------------------------------------------------------------
def interactive_tracking_and_labeling(video_path, output_dir, model_path, desired_fps=None,
                                      max_disp_w=900, max_disp_h=600):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Video could not be opened.")
        return 0  # no frames saved

    real_fps = cap.get(cv2.CAP_PROP_FPS)
    step, _ = calc_step(real_fps, desired_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    # label buffers
    gt = []
    absence = []
    cover = []
    cut = []

    target_class = None
    target_box = None
    tracking_active = False

    current_idx = 0
    save_idx = 1  
    print("[INFO] Start: W=Select, ENTER=Skip, A=Back, ESC=Exit")

    while not tracking_active and current_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[!] Read fail @ frame {current_idx}")
            break

        results = model(frame)[0]
        yolo_draw = []
        for b in results.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            yolo_draw.append((x1, y1, x2, y2))

        disp, _ = resize_for_display(frame, max_disp_w, max_disp_h)
        cv2.putText(disp, f"Frame {current_idx+1}/{total_frames} (step {step})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,255,50), 2)
        cv2.putText(disp, "W=Select  ENTER=Skip  A=Back  ESC=Exit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Selection Waiting", disp)
        key = cv2.waitKey(0) & 0xFF

        if key == 27: 
            print("[INFO] Exit during selection.")
            break

        elif key in (13, 10): 
            out_name = f"{save_idx:08d}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), frame)
            gt.append((0.0000,0.0000,0.0000,0.0000))
            absence.append(1)
            cover.append(0)
            cut.append(0)
            save_idx += 1
            current_idx += step
            continue

        elif key == ord('a'): 
            if save_idx <= 1:
                print("[i] Already at first sampled frame.")
                continue
            save_idx -= 1
            if gt: gt.pop()
            if absence: absence.pop()
            if cover: cover.pop()
            if cut: cut.pop()
            current_idx = max(0, current_idx - step)
            last_img = os.path.join(output_dir, f"{save_idx:08d}.jpg")
            if os.path.exists(last_img):
                os.remove(last_img)
            print(f"[INFO] Back to sampled frame {save_idx} (video frame idx {current_idx})")
            continue

        elif key == ord('w'):
            frame_scaled, scale = resize_for_display(frame, max_disp_w, max_disp_h)
            roi = interactive_roi_select_on_scaled(frame_scaled, scale, yolo_results=yolo_draw,
                                       win_name="ROI Select")

            if roi is None:
                print("[!] ROI not selected. Marking skipped.")
                out_name = f"{save_idx:08d}.jpg"
                cv2.imwrite(os.path.join(output_dir, out_name), frame)
                gt.append((0.0000,0.0000,0.0000,0.0000))
                absence.append(1)
                cover.append(0)
                cut.append(0)
                save_idx += 1
                current_idx += step
                continue

            best_i, best_det = 0.0, None
            for b in results.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                box = [x1, y1, x2 - x1, y2 - y1]
                sc = iou(roi, box)
                if sc > best_i:
                    best_i = sc
                    best_det = (int(b.cls), box)

            if best_det is None:
                print("[!] No matching det. Skip.")
                out_name = f"{save_idx:08d}.jpg"
                cv2.imwrite(os.path.join(output_dir, out_name), frame)
                gt.append((0.0000,0.0000,0.0000,0.0000))
                absence.append(1)
                cover.append(0)
                cut.append(0)
                save_idx += 1
                current_idx += step
                continue

            target_class, target_box = best_det
            tracking_active = True
            print(f"[INFO] Tracking start @ video frame {current_idx} class={target_class} box={target_box}")

            out_name = f"{save_idx:08d}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), frame)
            x,y,w,h = target_box
            gt.append((x,y,w,h))
            absence.append(0)
            cover.append(0)
            cut.append(1 if is_box_out_of_bounds(x,y,w,h,frame.shape[1],frame.shape[0]) else 0)
            save_idx += 1
            current_idx += step
            break

    if not tracking_active:
        write_labels_train_val(output_dir, gt, absence, cover, cut)
        cap.release()
        cv2.destroyAllWindows()
        return save_idx - 1

    # --------------------------------------------------------
    # Tracking loop: play forward, sample by step, auto-label
    # --------------------------------------------------------
    while current_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ok, frame = cap.read()
        if not ok:
            break

        model_frame = frame.copy()
        # YOLO detect
        results = model(model_frame)[0]
        matched = None
        best_i = 0.0
        for b in results.boxes:
            cls = int(b.cls)
            if cls != target_class:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            box = [x1, y1, x2 - x1, y2 - y1]
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
            cut.append(1 if is_box_out_of_bounds(x,y,w,h,model_frame.shape[1],model_frame.shape[0]) else 0)
            cv2.rectangle(model_frame,(x,y),(x+w,y+h),(0,0,255),2)
        else:
            gt.append((0,0,0,0))
            absence.append(1)
            cover.append(0)
            cut.append(0)

        out_name = f"{save_idx:08d}.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), frame)
        save_idx += 1
        
        display_frame = model_frame  # etiket çizmek istemiyorsan dokunma
        disp, _ = resize_for_display(display_frame, max_disp_w, max_disp_h)
        cv2.imshow("Tracking", disp)
        if cv2.waitKey(1) & 0xFF == 27: 
            print("[INFO] Tracking stopped early by user.")
            break

        current_idx += step

    write_labels_train_val(output_dir, gt, absence, cover, cut)
    cap.release()
    cv2.destroyAllWindows()
    return save_idx - 1

# ------------------------------------------------------------
# TEST split: interactive selection w/ YOLO + frame extract
# ------------------------------------------------------------
def interactive_selection_test(video_path, output_dir, model, desired_fps=None,
                               max_disp_w=900, max_disp_h=600):
   
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Video could not be opened:", video_path)
        return None, -1

    real_fps = cap.get(cv2.CAP_PROP_FPS)
    step, _ = calc_step(real_fps, desired_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO overlay
        results = model(frame)[0]
        yolo_draw = []
        for b in results.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            yolo_draw.append((x1, y1, x2, y2))

        disp, scale = resize_for_display(frame, max_disp_w, max_disp_h)
        for (x1, y1, x2, y2) in yolo_draw:
            sx1 = int(x1 * scale)
            sy1 = int(y1 * scale)
            sx2 = int(x2 * scale)
            sy2 = int(y2 * scale)
            cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

        cv2.putText(disp, f"Frame {frame_idx+1}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2)
        cv2.putText(disp, "[ENTER]=Select/Next  [A]=Back  [ESC]=Exit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Test Selection", disp)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # ESC
            print("[INFO] Çıkış yapıldı.")
            cap.release()
            cv2.destroyAllWindows()
            return None, -1
        elif k in (13, 10):  # ENTER
            frame_idx += step
            continue
        elif k == ord('a'):
            frame_idx = max(0, frame_idx - step)
            continue
        elif k == ord('w'):  # ROI seçim modu
            # ölçekli pencerede seçim
            roi = interactive_roi_select_on_scaled(
                disp, scale, yolo_results=yolo_draw,
                win_name="Test ROI Select", crosshair=True
            )
            if roi is not None:
                # ROI seçildi -> bitir
                cap.release()
                cv2.destroyWindow("Test Selection")
                return roi, frame_idx
            else:
                # seçim iptal edilirse aynı frame üzerinde kal veya ileri git?
                # Kullanıcı seçim yapmadıysa doğal akış: sonraki frame
                frame_idx += step
                continue

        else:
            # bilinmeyen tuş -> ileri
            frame_idx += step
                

    cap.release()
    return None, -1

def extract_sampled_frames(video_path, output_dir, start_frame_idx=0,
                           desired_fps=None):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    real_fps = cap.get(cv2.CAP_PROP_FPS)
    step, _ = calc_step(real_fps, desired_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = start_frame_idx
    save_idx = 1
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        out_name = f"{save_idx:08d}.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), frame)
        save_idx += 1
        frame_idx += step

    cap.release()
    return save_idx - 1

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main(args):
    split_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(split_dir, exist_ok=True)

    # load model once
    model = YOLO(args.model)

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
            print(f"[{args.split.upper()}] Processing {file} ...")
            saved = interactive_tracking_and_labeling(
                video_path, out_path, model_path=args.model, desired_fps=args.fps
            )
            print(f"[✓] {file} → {args.split}/{out_name} ({saved} sampled frame)")

        else:  # test
            print(f"[TEST] Selection mode for {file} ...")
            roi, selected_frame = interactive_selection_test(
                video_path, out_path, model, desired_fps=args.fps
            )
            # Now sample frames (from selected_frame forward)
            saved = extract_sampled_frames(
                video_path, out_path, start_frame_idx=max(selected_frame, 0),
                desired_fps=args.fps
            )
            if roi is not None and selected_frame >= 0:
                # clamp box to frame
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    fh, fw = frame.shape[:2]
                    x,y,w,h = roi
                    x = max(0, min(x, fw-1))
                    y = max(0, min(y, fh-1))
                    w = max(1, min(w, fw - x))
                    h = max(1, min(h, fh - y))
                    write_labels_test(out_path, x, y, w, h)
                else:
                    write_lines(os.path.join(out_path,"groundtruth.txt"), ["\n"])
                print(f"[✓] ROI saved ({roi}) → {out_name} | Sampled {saved} frame")
            else:
                # no ROI chosen → empty GT line (valid GOT-10k style fallback)
                write_lines(os.path.join(out_path,"groundtruth.txt"), ["\n"])
                print(f"[!] ROI not selected for {file}; GT empty. Sampled {saved} frame")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Video directory")
    parser.add_argument("--output_dir", required=True, help="GOT10k main directory")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--model", default="./model_weights/best.pt", help="YOLO model path")
    parser.add_argument("--fps", type=float, default=None, help="Desired FPS for frame sampling/labeling")
    args = parser.parse_args()

    main(args)
