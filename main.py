# main.py - SADELEŞTİRİLMİŞ FİNAL SÜRÜM (LLM YOK)
from fastapi import FastAPI, UploadFile, File, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64 
import os
import io
import cv2 
import numpy as np 
import shutil

# --- GÜVENLİK AYARLARI ---
API_KEY = "aygaz_secret_2025" 
API_KEY_NAME = "x-api-key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403, 
            detail=" GEÇERSİZ API ANAHTARI! Giriş reddedildi."
        )

# --- MODEL VE AYARLAR ---
TRAINED_MODEL_PATH = 'runs/detect/hard_hat_run/weights/best.pt'
CONFIDENCE_LEVEL = 0.25 

yolo_model = None
try:
    if os.path.exists(TRAINED_MODEL_PATH):
        yolo_model = YOLO(TRAINED_MODEL_PATH)
except Exception:
    pass

# --- Pydantic Modelleri (LLM Raporu Çıkarıldı) ---
class SafetyReportResponse(BaseModel):
    status: str
    message: str
    detections: dict
    # llm_report satırı silindi
    visual_output_b64: str | None = None 

app = FastAPI(title="Aygaz Güvenlik API (Secured)")

# =======================================================================
# 1. RESİM ANALİZİ (LLM SİZ)
# =======================================================================
@app.post("/api/v1/analyze_image", response_model=SafetyReportResponse)
async def analyze_image(
    file: UploadFile = File(...),
    api_key: str = Security(get_api_key)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir resim dosyası yükleyin.")
    
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    
    detection_counts = {}
    detected_person = 0
    visual_b64 = None

    if yolo_model:
        results = yolo_model.predict(source=img, conf=CONFIDENCE_LEVEL, save=False, device='0') 
        
        for r in results:
            for box in r.boxes:
                name = yolo_model.names[int(box.cls)]
                detection_counts[name] = detection_counts.get(name, 0) + 1
                if name in ['head', 'helmet', 'person']:
                    detected_person += 1
            
            im_np = r.plot() 
            im_bgr = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR) 
            is_success, buffer = cv2.imencode(".jpg", im_bgr)
            io_buf = io.BytesIO(buffer)
            visual_b64 = base64.b64encode(io_buf.getvalue()).decode('utf-8')
            break 
    
    # LLM Raporu oluşturma kısmı SİLİNDİ.

    return SafetyReportResponse(
        status="SUCCESS" if yolo_model else "PENDING_TRAINING",
        message=f"{detected_person} kişi tespit edildi.",
        detections=detection_counts,
        visual_output_b64=visual_b64
    )

# =======================================================================
# 2. VİDEO ANALİZİ (Aynı Kalıyor)
# =======================================================================
@app.post("/api/v1/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    api_key: str = Security(get_api_key)
):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir video dosyası yükleyin.")

    if not yolo_model:
         raise HTTPException(status_code=500, detail="Model yüklenemedi.")

    temp_input = "temp_input_video.mp4"
    temp_output = "processed_output_video.mp4"
    
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(temp_input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        
        results = yolo_model.predict(source=frame, conf=CONFIDENCE_LEVEL, save=False, device='0', verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    
    return FileResponse(temp_output, media_type="video/mp4", filename="guvenlik_analizi.mp4")