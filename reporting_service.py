# reporting_service.py
from transformers import pipeline

# Türkçeyi destekleyen model (İlk çalıştırmada otomatik indirilir)
try:
    report_generator = pipeline("text-generation", model="dbmdz/bert-base-turkish-cased")
except Exception:
    report_generator = pipeline("text-generation", model="gpt2")

def generate_safety_report(yolo_results: dict) -> str:
    """YOLO sonuçlarını alıp LLM kullanarak resmi bir rapor oluşturur."""
    
    helmet_count = yolo_results.get("helmet", 0)
    head_count = yolo_results.get("head", 0)
    total_person = yolo_results.get("person", yolo_results.get("total", 0))

    if total_person == 0:
        return "Görüntüde personel tespit edilmemiştir. Saha güvenli."
    
    # LLM'ye göndereceğimiz komut (Prompt)
    prompt = (
        f"Aygaz tesis güvenliği raporu. Alanda toplam {total_person} kişi var. "
        f"{head_count} kişi baret takmamaktadır. "
        f"{helmet_count} kişi ise baret takmaktadır. "
        "Bu verilere dayanarak, tesis yöneticisine hitaben kısa, resmi bir güvenlik raporu özeti hazırla. "
        "Özette baret takmayan personelin güvenliğinin tehlikede olduğu vurgulanmalı. Raporun başlığını 'ACİL GÜVENLİK BİLDİRİMİ' olarak belirle."
    )
    
    # Raporu Üret
    response = report_generator(prompt, max_length=150, num_return_sequences=1, truncation=True)
    return response[0]['generated_text']