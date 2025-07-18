# GOT-10k Veri Seti Hazırlama ve Etiketleme Araçları

Gerekli kütüphaneleri yükleyerek başlayalım:

```bash
pip install -r "requirements.txt"
```

---

## 1) convert\_to\_got10k.py

Verdiğiniz klasör dizini içerisindeki videoların kendi fps değerlerine göre framelere ayrılmasını ve got10k uygun dizin dosya yapısının kurulmasını sağlar. ".mp4", ".avi", ".mov", ".mkv" video formatları desteklenir.

GOT-10k veriseti aşağıdaki dizin yapısına sahiptir. --split ile vereceğiniz değere göre videolarınız framelere ayrılacak ve gerekli dosyalar oluşturulacaktır.

```
GOT-10k/
├── train/
│   ├── [class_name]_[video_id]/
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
│   │   ├── groundtruth.txt
│   │   ├── absence.label
│   │   ├── cover.label
│   │   └── cut_by_image.label
│   └── ...
│
├── val/
│   ├── [class_name]_[video_id]/
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
│   │   ├── groundtruth.txt
│   │   ├── absence.label
│   │   ├── cover.label
│   │   └── cut_by_image.label
│   └── ...
│
├── test/
│   ├── [class_name]_[video_id]/
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
│   │   └── groundtruth.txt  ← sadece ilk bbox (1 satır)
│   └── ...
```

### Çalıştırma Komutu:

```bash
python convert_to_got10k.py --input_dir ./data --output_dir ./got10k_dataset --split test --model ./model_weights/best.pt --fps 20
```

### \--input\_dir -> Got10k formatına çevirmek istediğiniz videolarınızın bulunduğu dizin
### \--output\_dir -> Got10k formatında verilerinizin kaydedileceği dizin
### \--split -> Verilerinizi dönüştürmek istediğiniz format (test, train, val)
### \--model -> Nesne tespiti için kullanılacak yolo modeli
### \--fps -> Saniyede işlenecek kare sayısı



\--split train ya da val seçmeniz durumunda karşınıza seçtiğiniz videonun ilk karesi açılacak. W tuşuna bastıktan sonra buradan mouse ile takip etmek istediğiniz nesneyi seçmeniz gerekli. Seçim yaptığınız bölgeyi kapsayan en yakın bbox değeri tracker ile takip edilmeye başlanacak. 

Tracker'ın her frame sonucunda aldığı bbox değeri train/car\_1/groundtruth.txt dosyasında ilgili satıra yazılacak. Bu sayede yolo modeli ile beraber tracker kullanarak verileri hızlı bir şekilde etiketlemiş olacağız.

Eğer takip etmek istediğiniz nesne açılan ilk framede değilse enter tuşuna basarak nesneyi görene kadar ilerleyebilirsiniz. Etiket işlemini yaptıktan sonra tracker başlayacak ve video akacaktır. Geri gelmek isterseniz A tuşunu kullanabilirsiniz.

\--split test seçmeniz durumunda dizinini verdiğiniz klasördeki videolar karelere ayrılırken karşınıza ilk kare açılacak ve takip edilmesini istediğiniz nesnenin seçilmesi istenecek. Seçim yaptıktan sonra enter tuşuna bastığınızda bbox değerleri test/video\_1/groundtruth.txt dosyasına kaydedilecektir.

Eğer takip etmek istediğiniz nesne ilk karede yok ise seçim yapmadan enter tuşuna basarak ilerleyebilirsiniz. Nesne kareye girdiğinde seçim yaparak enter tuşuna bastığınızda bbox değerleri aynı şekilde groundtruth.txt dosyasına kaydedilecektir. Geri gelmek isterseniz A tuşunu kullanabilirsiniz.

### absence.label -> Takip edilecek nesnenin kare içerisinde olup (0) olmadığı (1) bilgisini her kare için bir satırda saklar.
### cover.label -> Takip edilecek nesnenin önünde engel olup (1) olmadığı (0) bilgisini saklar.
### cut\_by\_image.label -> Takip edilecek nesnenin görüntü karesinden taşıp (1) taşmadığı (0) bilgisini saklar.

---

## 2) visualize\_groundtruth.py

Bu dosya ile bir önceki adımda groundtruth.txt dosyasına kaydettiğiniz bbox değerlerini frameler üzerinde görebilirsiniz.
Güncellemek istediğiniz label için önce C tuşuna basarak bbox temizleyin. Sonrasında W tuşuna basarak seçim yapabilirsiniz. Seçiminiz bittiğinde enter tuşuna basarsanız bbox değerleriniz groundtruth.txt dosyasından ve buna bağlı diğer .label dosyalarındaki ilgili satırlardan güncellenecektir. Seçim yapmak istemiyorsanız enter basarak ilerleyebilirsiniz. Geri gelmek için A tuşunu kullanabilirsiniz.

### Çalıştırma Komutu:

```bash
python visualize_groundtruth.py --images_dir ./got10k_dataset/train/car_1/
```

---

## 3) label_cover.py
cover.label dosyasının içeriğini oluşturacağımız bu dosyayı çalıştırdığınızda karşınıza nesne tespiti sonucu etiketlenen bbox görselleri açılacak. Nesne önünde engel var ise 0 yok ise 1 tuşuna basarak cover.label dosyasının verisini oluşturabilirsiniz. Geri gitmek için A tuşunu kullanabilirsiniz.

### Çalıştırma Komutu:

```bash
python label_cover.py --images_dir ./got10k_dataset/train/car_1```
```

---

## 4) yolo2got10k.py

Elinizde hazır yolo formatında etiketlenmiş ve içinde her frame için bir .txt dosyası bulunduran labels klasörünüz varsa bu dosya yardımıyla bbox bilgisini içeren tek bir groundtruth.txt dosyası altında verilerinizi toplayabilirsiniz.
## NOT: GOT10K tekli nesne tespit verisetidir. Bu dosya için vereceğiniz labels klasörü altındaki txt dosyalarında tek satır bbox bilgisi olmalıdır. Birden fazla etiket bilgisi olması durumunda hata alabilirsiniz.

### Çalıştırma Komutu:

```bash
python yolo2got10k.py --yolo_dir ./path/to/your/labels/ --img_width 1280 --img_height 720 --output_file ./got10k_dataset/train/car_1/groundtruth.txt
```

#### Yolo normalize edilmiş bbox verisi kullanırken got10k normalize edilmemiş bbox verisi kullanır. Bu yüzden burada verilecek img\_width ve img\_height değerleri görselin orijinal değerleri olmalıdır.
