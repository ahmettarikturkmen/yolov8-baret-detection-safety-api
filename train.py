from ultralytics import YOLO # ultralytics kütüphanesinden YOLO sınıfı

DATASET_PATH = 'data.yaml' 

# Bu yapı, Windows'taki çoklu işlem Multiprocessing için.
if __name__ == '__main__':
    
    # Başlangıç modeli: yolov8s.pt (YOLOv8 small)-Transfer Learning
    model = YOLO('yolov8s.pt')
    
    # Eğitim için sonuçların nereye kaydedileceğini belirle
    PROJECT_NAME = 'runs/detect' 
    
    print(f"Eğitim başlıyor. Veri yolu: {DATASET_PATH}")
    
    # Eğitimi başlatıyoruz
    results = model.train(
        data=DATASET_PATH, 
        epochs=50, 
        imgsz=416, 
        name='hard_hat_run', # eğitim ismi
        project=PROJECT_NAME 
    )
    
    # En iyi modelin kaydedileceği yol
    print("\nEğitim tamamlandı! En iyi model 'runs/detect/hard_hat_run/weights/best.pt' yoluna kaydedildi.")