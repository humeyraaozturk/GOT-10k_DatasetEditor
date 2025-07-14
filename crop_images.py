import cv2
import os
from glob import glob

# === AYARLAR ===
input_folder = 'dataset/uai/test/good'
output_folder = 'dataset/uai/test/good'
target_width = 800
target_height = 600

os.makedirs(output_folder, exist_ok=True)
image_paths = sorted(glob(os.path.join(input_folder, '*.jpg')))

# === GLOBAL DEĞİŞKENLER ===
cursor_x, cursor_y = -1, -1
selecting = False
ix = iy = fx = fy = -1

# === MOUSE CALLBACK ===
def mouse_draw(event, x, y, flags, param):
    global cursor_x, cursor_y, selecting, ix, iy, fx, fy
    cursor_x, cursor_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        ix, iy = x, y
        fx, fy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        fx, fy = x, y

# === GÖRSELLERİ DÖNGÜYLE İŞLE ===
for i, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    if img is None:
        print(f"{img_path} okunamadı, atlanıyor.")
        continue

    print(f"📷 Şu an: {os.path.basename(img_path)}")

    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
    clone = resized_img.copy()

    # Seçim değerlerini resetle
    ix = iy = fx = fy = -1

    winname = "Crop Tool (Çiz, Enter → Kırp, C → Temizle)"
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, mouse_draw)

    while True:
        display = clone.copy()

        # Kılavuz çizgiler
        if 0 <= cursor_x < display.shape[1] and 0 <= cursor_y < display.shape[0]:
            cv2.line(display, (cursor_x, 0), (cursor_x, display.shape[0]), (0, 255, 0), 1)
            cv2.line(display, (0, cursor_y), (display.shape[1], cursor_y), (0, 255, 0), 1)

        # Seçim kutusu
        if ix != -1 and iy != -1 and fx != -1 and fy != -1 and (selecting or (ix != fx and iy != fy)):
            cv2.rectangle(display, (ix, iy), (fx, fy), (255, 0, 0), 2)

        cv2.imshow(winname, display)
        key = cv2.waitKey(10)

        if key == 13:  # Enter
            if ix != fx and iy != fy:
                x1 = int(min(ix, fx) / scale)
                y1 = int(min(iy, fy) / scale)
                x2 = int(max(ix, fx) / scale)
                y2 = int(max(iy, fy) / scale)
                
                
                
                cropped = img[y1:y2, x1:x2]
                output_path = os.path.join(output_folder, os.path.basename(img_path))
                cv2.imwrite(output_path, cropped)
                print(f"✅ KAYDEDİLDİ → {output_path}")
                break
            else:
                print("⛔ Geçerli bir seçim yapılmadı.")
                break

        elif key == ord('c'):
            print("🧼 Seçim temizlendi, tekrar deneyebilirsin.")
            ix = iy = fx = fy = -1
            selecting = False

    cv2.destroyAllWindows()

print("🎉 Tüm görseller başarıyla işlendi!")
