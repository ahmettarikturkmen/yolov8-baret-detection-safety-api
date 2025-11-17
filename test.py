from ultralytics import YOLO
import os

# --- MODEL VE RESİM YOLLARI ---
# best.pt dosyasının oluştuğu klasör yolunu kontrol et
TRAINED_MODEL_PATH = 'runs/detect/hard_hat_run/weights/best.pt'
TEST_IMAGE = 'test_image.jpg' 

# -------------------------------

if __name__ == '__main__':
    print("Model yükleniyor...")
    
    # 1. Modeli Yükle
    try:
        model = YOLO(TRAINED_MODEL_PATH)
    except FileNotFoundError:
        print(f"HATA: Eğitilmiş model dosyası '{TRAINED_MODEL_PATH}' bulunamadı.")
        print("Lütfen eğitimin bittiğinden ve yolun doğru olduğundan emin olun.")
        exit()

    print(f"Model başarıyla yüklendi. Test resmine tahmin yapılıyor: {TEST_IMAGE}")

    # 2. Tahmin Yap ve Sonucu Kaydet
    # conf=0.5: Güven skoru %50'nin üzerindeki tespitleri göster
    # save=True: Sonucu 'runs/detect/predict...' klasörüne kareli olarak kaydeder
    results = model.predict(source=TEST_IMAGE, conf=0.5, save=True, show=False, device='0')

    # 3. Sonuçları Terminalde Göster
    for r in results:
        boxes = r.boxes
        print(f"\n--- TESPİT SONUÇLARI ---")
        print(f"Tahmin edilen nesne sayısı: {len(boxes)}")
        
        for box in boxes:
            cls = int(box.cls)
            name = model.names[cls]
            conf = box.conf.item() * 100
            print(f"- Tespit edilen: {name}, Güven: {conf:.2f}%")

    print("\nTest tamamlandı!")
    print("Karelenmiş resim sonucu, 'runs/detect/predict...' klasörüne kaydedildi.")