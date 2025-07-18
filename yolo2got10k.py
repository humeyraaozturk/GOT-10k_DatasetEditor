import os
import argparse

def yolo_to_got10k(x_center, y_center, w, h, img_w, img_h):
    x = (x_center * img_w) - (w * img_w) / 2
    y = (y_center * img_h) - (h * img_h) / 2
    width = w * img_w
    height = h * img_h
    return x, y, width, height

def main(yolo_dir, img_width, img_height, output_file):
    files = sorted([f for f in os.listdir(yolo_dir) if f.endswith('.txt')])

    with open(output_file, 'w') as out_f:
        for file_name in files:
            path = os.path.join(yolo_dir, file_name)
            with open(path, 'r') as f:
                line = f.readline().strip()
                if not line:
                    # Dosya boşsa 0 bbox yaz
                    out_f.write("0.0000,0.0000,0.0000,0.0000\n")
                    continue

                parts = line.split()
                if len(parts) < 5:
                    print(f"[!] Warning: '{file_name}' Unexpected format")
                    out_f.write("0.0000,0.0000,0.0000,0.0000\n")
                    continue

                _, x_center, y_center, w, h = parts
                x_center, y_center, w, h = map(float, (x_center, y_center, w, h))
                x, y, width, height = yolo_to_got10k(x_center, y_center, w, h, img_width, img_height)

                out_f.write(f"{x:.4f},{y:.4f},{width:.4f},{height:.4f}\n")

    print(f"[✓] Translation completed, Output: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_dir", required=True, help="YOLO labels directory")
    parser.add_argument("--img_width", type=int, required=True, help="Image width (pixels)")
    parser.add_argument("--img_height", type=int, required=True, help="Image height (pixels)")
    parser.add_argument("--output_file", required=True, help="Output groundtruth file path")
    args = parser.parse_args()

    main(args.yolo_dir, args.img_width, args.img_height, args.output_file)
