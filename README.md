# 🏠 Ev Fiyat Tahmin Projesi

Ames, Iowa konutlarının satış fiyatını makine öğrenmesiyle tahmin eden uçtan uca bir proje.

**Veri:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) · MIT Lisansı

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Notebook'u çalıştır
Jupyter'da `ev_fiyat_tahmin.ipynb` dosyasını aç → Kernel → Restart & Run All

### 2. Streamlit arayüzünü başlat
```bash
streamlit run app.py
```

## Proje Yapısı

```
ev_fiyat_tahmin/
├── ev_fiyat_tahmin.ipynb   # Tüm analiz ve modelleme
├── app.py                  # Streamlit arayüzü
├── requirements.txt        # Bağımlılıklar
├── train.csv               # Eğitim verisi
├── test.csv                # Test verisi
├── submission.csv          # Kaggle submission
└── model/
    ├── final_model.pkl     # Eğitilmiş model
    └── feature_names.pkl   # Özellik isimleri
```

## Proje Aşamaları

| # | Aşama |
|---|-------|
| 1 | Problem Tanımı |
| 2 | Veri Yükleme |
| 3 | Keşifsel Veri Analizi (EDA) |
| 4 | Veri Kalitesi |
| 5 | Preprocessing & Feature Engineering |
| 6 | Train/Val Split |
| 7 | Baseline Model |
| 8 | Model Karşılaştırması (6 model) |
| 9 | Aday Model Seçimi |
| 10 | Hiperparametre Optimizasyonu |
| 11 | Strong Competitor Karşılaştırması |
| 12 | Final Değerlendirme |
| 13 | Submission |

## Sonuçlar

- **Final Model:** LightGBM (Tuned)
- **RMSE:** ~0.12 (log ölçek)
- **R²:** ~0.92
- **Baseline iyileşmesi:** ~%69

## Teknolojiler

Python · Pandas · Scikit-learn · LightGBM · XGBoost · Streamlit · Matplotlib · Seaborn
