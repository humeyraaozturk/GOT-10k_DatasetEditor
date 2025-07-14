"""
python yolo2got10k.py --yolo_dir ./got10k_dataset/train/human_1/labels/ --img_width 1280 --img_height 720 --output_file ./got10k_dataset/train/human_1/groundtruth.txt
"""
import os
import argparse

def yolo_to_got10k(x_center, y_center, w, h, img_w, img_h):
    # YOLO: normalize edilmiş merkez, genişlik, yükseklik
    # GOT10k: pixel bazlı sol üst x,y, genişlik, yükseklik
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
                    out_f.write("0 0 0 0\n")
                    continue

                parts = line.split()
                if len(parts) < 5:
                    print(f"[!] Uyarı: '{file_name}' içinde beklenmedik format")
                    out_f.write("0 0 0 0\n")
                    continue

                # YOLO format: class_id x_center y_center width height
                _, x_center, y_center, w, h = parts
                x_center, y_center, w, h = map(float, (x_center, y_center, w, h))
                x, y, width, height = yolo_to_got10k(x_center, y_center, w, h, img_width, img_height)

                out_f.write(f"{x:.2f} {y:.2f} {width:.2f} {height:.2f}\n")

    print(f"[✓] Dönüşüm tamamlandı, çıktı: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_dir", required=True, help="YOLO etiketlerinin olduğu klasör")
    parser.add_argument("--img_width", type=int, required=True, help="Görüntü genişliği (pixel)")
    parser.add_argument("--img_height", type=int, required=True, help="Görüntü yüksekliği (pixel)")
    parser.add_argument("--output_file", required=True, help="Çıktı groundtruth dosya yolu")
    args = parser.parse_args()

    main(args.yolo_dir, args.img_width, args.img_height, args.output_file)
