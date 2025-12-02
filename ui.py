# ui.py - KULLANICI ARAYÜZÜ (LLM YOK, SADECE TESPİT)
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# --- AYARLAR ---
API_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="Aygaz Güvenlik Sistemi", 
    page_icon="👷", 
    layout="wide"
)

# --- BAŞLIK ---
st.title(" YZ Destekli Baret Tespit Sistemi")
st.markdown("""
Bu panel, sahadan gelen görüntüleri **YOLOv8** ile analiz eder ve güvenlik ihlallerini görselleştirir.
""")

# --- YAN MENÜ ---
with st.sidebar:
    st.header(" Yetkili Girişi")
    api_key = st.text_input("API Anahtarı (x-api-key)", type="password")
    st.info(" Şifre: `aygaz_secret_2025`")
    st.divider()
    st.write("© 2025 Aygaz Ar-Ge Aday Projesi")

# --- DOSYA YÜKLEME ---
uploaded_file = st.file_uploader(
    "Analiz için Resim veya Video Yükleyin", 
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi']
)

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    st.write("---")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(" Yüklenen Dosya")
        if file_type == 'image':
            st.image(uploaded_file, use_container_width=True)
        elif file_type == 'video':
            st.video(uploaded_file)

    with col2:
        st.subheader(" İşlem Merkezi")
        analyze_btn = st.button(" Analizi Başlat", type="primary", use_container_width=True)

        if analyze_btn:
            if not api_key:
                st.error(" Lütfen sol menüden API Anahtarını giriniz!")
            else:
                headers = {"x-api-key": api_key}
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                status_box = st.status("Yapay Zeka çalışıyor...", expanded=True)
                
                try:
                    # --- RESİM ANALİZİ ---
                    if file_type == 'image':
                        status_box.write(" Resim API'ye gönderiliyor...")
                        response = requests.post(f"{API_URL}/analyze_image", headers=headers, files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            status_box.update(label="İşlem Başarılı!", state="complete", expanded=False)
                            
                            st.success(f"Sonuç: {result['message']}")
                            
                            # 1. KARELİ RESİM
                            if result['visual_output_b64']:
                                image_bytes = base64.b64decode(result['visual_output_b64'])
                                st.image(image_bytes, caption="Tespit Sonuçları", use_container_width=True)
                            
                            # 2. İSTATİSTİKLER (Rapor yerine sadece sayıları gösteriyoruz)
                            st.info(" **Tespit İstatistikleri**")
                            dets = result['detections']
                            m1, m2, m3 = st.columns(3)
                            m1.metric(" Baretli", dets.get('helmet', 0))
                            m2.metric(" Baretsiz", dets.get('head', 0))
                            m3.metric(" Toplam Kişi", dets.get('person', 0) + dets.get('head', 0) + dets.get('helmet', 0))
                            
                        elif response.status_code == 403:
                            status_box.update(label="Hata!", state="error")
                            st.error(" Yetkisiz Giriş! API Anahtarı yanlış.")
                        else:
                            st.error(f"Sunucu Hatası: {response.text}")

                    # --- VİDEO ANALİZİ ---
                    elif file_type == 'video':
                        status_box.write(" Video işleniyor... Bu işlem biraz sürebilir.")
                        response = requests.post(f"{API_URL}/analyze_video", headers=headers, files=files)
                        
                        if response.status_code == 200:
                            status_box.update(label="Video Hazır!", state="complete", expanded=False)
                            
                            output_filename = "sonuc_videosu.mp4"
                            with open(output_filename, "wb") as f:
                                f.write(response.content)
                            
                            st.success(" Video başarıyla işlendi!")
                            st.video(output_filename)
                            
                            with open(output_filename, "rb") as file:
                                st.download_button(
                                    label=" İşlenmiş Videoyu İndir",
                                    data=file,
                                    file_name="guvenlik_analizi.mp4",
                                    mime="video/mp4"
                                )
                        
                        elif response.status_code == 403:
                            status_box.update(label="Hata!", state="error")
                            st.error(" Yetkisiz Giriş! API Anahtarı yanlış.")
                        else:
                            st.error(f"Hata: {response.text}")

                except Exception as e:
                    status_box.update(label="Bağlantı Hatası", state="error")
                    st.error(f"API'ye ulaşılamadı.\nHata Detayı: {e}")