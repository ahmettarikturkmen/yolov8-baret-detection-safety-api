from ultralytics import YOLO
import os

# best.pt dosyasının oluştuğu klasör yolu
TRAINED_MODEL_PATH = 'runs/detect/hard_hat_run/weights/best.pt'
TEST_IMAGE = 'test_image.jpg' 

# Bu yapı, Windows'taki çoklu işlem Multiprocessing için.
if __name__ == '__main__':
    
    print("Model yükleniyor...")
    
    # Modeli Yükle
    try:
        model = YOLO(TRAINED_MODEL_PATH)
    except FileNotFoundError:
        print(f"HATA: Eğitilmiş model dosyası '{TRAINED_MODEL_PATH}' bulunamadı.")
        exit()

    print(f"Model başarıyla yüklendi. Test resmine tahmin yapılıyor: {TEST_IMAGE}")

    # Tahmin Yap ve Sonucu Kaydet
    # conf=0.25: Güven skoru %25'nin üzerindeki tespitleri göster
    # save=True: Sonucu 'runs/detect/predict...' klasörüne kareli olarak kaydeder
    # device:0 gpu kullanımı
    results = model.predict(source=TEST_IMAGE, conf=0.25, save=True, show=False, device='0')

    # Sonuçları Terminalde Göster
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