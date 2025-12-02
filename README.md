#  YZ Destekli Baret Tespit ve Güvenlik Sistemi
# YOLOv8 & FastAPI & Streamlit

##  Proje Tanımı
Bu proje, endüstriyel sahalarda iş güvenliğini artırmak amacıyla geliştirilmiş, **Yapay Zeka** tabanlı bir baret (hard hat) tespit sistemidir. 

Sistem, **YOLOv8** modelini kullanarak resim ve videolardaki çalışanları ve baret kullanım durumlarını anlık olarak tespit eder. **FastAPI** ile güvenli bir Backend servisi sunarken, **Streamlit** ile geliştirilen kullanıcı dostu arayüz üzerinden kolayca kontrol edilebilir.

##  Temel Yetenekler

* ** Güvenli API (Authentication):** API uç noktaları, yetkisiz erişimi engellemek için **API Key** korumasına sahiptir.
* ** Çoklu Medya Analizi:** Hem **Resim** (.jpg, .png) hem de **Video** (.mp4) dosyalarını analiz edebilir.
* ** Hassas Tespit:** YOLOv8 (Deep Learning) ile Baret, Baretsiz Kafa ve Kişi tespiti yapar.
* ** Web Arayüzü:** Streamlit tabanlı modern arayüz sayesinde kod yazmadan sistem kullanılabilir.
* ** Video İşleme:** Videoları kare kare işleyerek tespit kutularının çizildiği yeni bir video dosyası üretir.

##  Teknoloji Yığını (Tech Stack)

| Kategori | Teknoloji | Amaç |
| :--- | :--- | :--- |
| **Yapay Zeka** | **YOLOv8** | Nesne Tespiti (Object Detection) |
| **Backend** | **FastAPI** | Yüksek performanslı ve güvenli API servisi |
| **Frontend** | **Streamlit** | Kullanıcı Arayüzü (Web UI) |
| **Görüntü İşleme** | **OpenCV (cv2)** | Video işleme ve çizim işlemleri |
| **Dağıtım** | **NSSM** | API'yı Windows Servisi olarak sürekli çalıştırma |

