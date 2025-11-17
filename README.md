PROJE: Gelişmiş YOLOv8 modeli kullanılarak tasarlanmış, baret (hard hat) tespitine odaklanan tam kapsamlı bir güvenlik API'sıdır. FastAPI aracılığıyla görüntü ve video işleyebilir, tespit sonuçlarını Hugging Face LLM kullanarak resmi, doğal dil raporlarına dönüştürebilir.

Bu proje, Yapay Zeka ile endustriyel guvenlik uyumlulugunu (Baret/Kask kullanimi) denetleyen, tam kapsamli bir API cozumudur.

TEMEL YETENEKLER

* Coklu Medya Analizi: Resim ve video dosyalarini analiz edebilme.
* Anlik Tespit: YOLOv8 (Deep Learning) ile baret, baretsiz kafa ve kisi tespiti.
* Otomatik Raporlama: Tespit sonuclarini, LLM (Buyuk Dil Modeli) kullanarak resmi metin raporlarina donusturme.

TEKNOLOJI OZETI

Proje, is ilaninda belirtilen kritik teknolojilerin tamamini kullanmaktadir:

- YOLOv8: Nesne Tespiti (Goz)
- FastAPI: Yuksek Hizli API Catisi (Web API)
- Hugging Face: Dogal Dil Raporlama (LLM)
- OpenCV: Goruntu/Video Isleme
- Conda: Sanal Ortam Yonetimi (Anaconda)

API UC NOKTALARI (Endpoints)

- /analyze_image: Resim yukleme (Cikti: JSON Raporu + Base64 Kareli Resim).
- /analyze_video: Video yukleme (Cikti: Işlenmiş .mp4 Dosyasi).
- /docs: Swagger UI (API'yi tarayicida test etme arayuzu).

BASLANGIC KOMUTU

Proje klasorunde ortamınızı aktifleştirdikten sonra (Conda ile) bu komutla calistirin:

uvicorn main:app --reload