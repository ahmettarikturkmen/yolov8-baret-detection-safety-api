from ultralytics import YOLO

# ----------------------------------------------------------------------------------
# data.yaml dosyasının yolunu kontrol et
# train.py ve data.yaml aynı klasörde olduğu için sadece 'data.yaml' kullanıyoruz.
# ----------------------------------------------------------------------------------
DATASET_PATH = 'data.yaml' 

# Bu yapı, Windows'taki çoklu işlem hatasını çözer.
if __name__ == '__main__':
    
    # Başlangıç modeli: yolov8s.pt (YOLOv8 small - ilk çalıştırmada otomatik indirilir)
    model = YOLO('yolov8s.pt')
    
    # Eğitim için sonuçların nereye kaydedileceğini belirle
    PROJECT_NAME = 'runs/detect' 
    
    print(f"Eğitim başlıyor. Veri yolu: {DATASET_PATH}")
    
    # Eğitimi başlatıyoruz
    # Model artık GPU üzerinde çalışacaktır.
    results = model.train(
        data=DATASET_PATH, 
        epochs=50, 
        imgsz=416, 
        name='hard_hat_run', # Bu eğitime verilen isim
        project=PROJECT_NAME # Ana proje klasörünün adı
    )
    
    # En iyi modelin kaydedileceği yolu kullanıcıya bildir
    print("\nEğitim tamamlandı! En iyi model 'runs/detect/hard_hat_run/weights/best.pt' yoluna kaydedildi.")