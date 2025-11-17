# main.py - RESİM VE VİDEO DESTEKLİ FİNAL SÜRÜM
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse # Video indirmek için FileResponse gerekli
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64 
import os
import io
import cv2 
import numpy as np 
import shutil # Video dosyalarını kaydetmek için
from reporting_service import generate_safety_report 

# --- MODEL YÜKLEME ---
TRAINED_MODEL_PATH = 'runs/detect/hard_hat_run/weights/best.pt'
yolo_model = None
try:
    if os.path.exists(TRAINED_MODEL_PATH):
        yolo_model = YOLO(TRAINED_MODEL_PATH)
except Exception:
    pass

# --- Pydantic Modelleri ---
class SafetyReportResponse(BaseModel):
    status: str
    message: str
    detections: dict
    llm_report: str
    visual_output_b64: str | None = None 

class Base64Input(BaseModel):
    base64_data: str 

app = FastAPI(title="Aygaz Tesis Güvenlik ve Raporlama API")

# =======================================================================
# 1. RESİM ANALİZİ (Rapor + Kareli Resim)
# =======================================================================
@app.post("/api/v1/analyze_image", response_model=SafetyReportResponse)
async def analyze_image(file: UploadFile = File(...)):
    
    # Sadece resim dosyalarını kabul et
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir resim dosyası yükleyin.")
    
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    
    detection_counts = {}
    detected_person = 0
    visual_b64 = None

    if yolo_model:
        # Resim üzerinde tahmin yap
        results = yolo_model.predict(source=img, conf=0.60, save=False, device='0') 
        
        for r in results:
            # Sayım işlemi
            for box in r.boxes:
                name = yolo_model.names[int(box.cls)]
                detection_counts[name] = detection_counts.get(name, 0) + 1
                if name in ['head', 'helmet', 'person']:
                    detected_person += 1
            
            # Kareli resmi çiz ve Base64'e çevir
            im_np = r.plot() 
            im_bgr = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR) 
            is_success, buffer = cv2.imencode(".jpg", im_bgr)
            io_buf = io.BytesIO(buffer)
            visual_b64 = base64.b64encode(io_buf.getvalue()).decode('utf-8')
            break 
            
    # LLM Raporu Oluşturma
    yolo_data = {
        "helmet": detection_counts.get("helmet", 0),
        "head": detection_counts.get("head", 0),
        "person": detected_person or 1, 
        "total": detected_person or 1
    }
    report_text = generate_safety_report(yolo_data)

    return SafetyReportResponse(
        status="SUCCESS" if yolo_model else "PENDING_TRAINING",
        message=f"{detected_person} kişi tespit edildi." if yolo_model else "Model henüz hazır değil.",
        detections=detection_counts,
        llm_report=report_text,
        visual_output_b64=visual_b64
    )

# =======================================================================
# 2. VİDEO ANALİZİ (Kareli Video İndirme)
# =======================================================================
@app.post("/api/v1/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    
    # Sadece video dosyalarını kabul et
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir video dosyası (mp4, avi) yükleyin.")

    if not yolo_model:
         raise HTTPException(status_code=500, detail="Model yüklenemedi.")

    # Geçici dosya yolları
    temp_input = "temp_input_video.mp4"
    temp_output = "processed_output_video.mp4"
    
    # 1. Yüklenen videoyu diske kaydet
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Videoyu OpenCV ile aç
    cap = cv2.VideoCapture(temp_input)
    
    # Video özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video kaydediciyi ayarla (MP4 formatı için 'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # 3. Kare kare işleme
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Video bitti
        
        # YOLO tahmini yap (tek bir kare üzerinde)
        results = yolo_model.predict(source=frame, conf=0.2, save=False, device='0', verbose=False)
        
        # Kare üzerine kutuları çiz
        annotated_frame = results[0].plot()
        
        # İşlenmiş kareyi yeni videoya ekle
        out.write(annotated_frame)

    # 4. Temizlik
    cap.release()
    out.release()
    
    # 5. İşlenmiş videoyu geri gönder
    return FileResponse(temp_output, media_type="video/mp4", filename="aygaz_guvenlik_analizi.mp4")

# =======================================================================
# 3. GÖRSELLEŞTİRME (Base64 -> Resim)
# =======================================================================
@app.post("/api/v1/visualize_b64")
async def visualize_b64(b64_data: Base64Input):
    try:
        base64_str = b64_data.base64_data 
        if base64_str:
            img_bytes = base64.b64decode(base64_str)
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            is_success, buffer = cv2.imencode(".jpg", img)
            return Response(content=buffer.tobytes(), media_type="image/jpeg")  
        return {"detail": "No Base64 data provided"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Görselleştirme hatası: {e}")