"""
🏠 Ev Fiyat Tahmin Uygulaması
Kaggle — House Prices: Advanced Regression Techniques
Streamlit Arayüzü
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Sayfa Ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 Ev Fiyat Tahmini",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────
ORDINAL_MAP = {
    'ExterQual'   : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0},
    'ExterCond'   : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0},
    'BsmtQual'    : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0},
    'BsmtCond'    : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0},
    'BsmtExposure': {'Gd':4,'Av':3,'Mn':2,'No':1,'None':0},
    'BsmtFinType1': {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0},
    'BsmtFinType2': {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0},
    'HeatingQC'   : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},
    'KitchenQual' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0},
    'FireplaceQu' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0},
    'GarageQual'  : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0},
    'GarageCond'  : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0},
    'GarageFinish': {'Fin':3,'RFn':2,'Unf':1,'None':0},
    'PoolQC'      : {'Ex':4,'Gd':3,'TA':2,'Fa':1,'None':0},
    'Fence'       : {'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'None':0},
    'PavedDrive'  : {'Y':2,'P':1,'N':0},
    'CentralAir'  : {'Y':1,'N':0},
    'Street'      : {'Pave':1,'Grvl':0},
    'LandSlope'   : {'Gtl':2,'Mod':1,'Sev':0},
    'Functional'  : {'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0},
}

NONE_COLS = [
    'PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
    'GarageType','GarageFinish','GarageQual','GarageCond',
    'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
    'MasVnrType'
]
ZERO_COLS = [
    'GarageYrBlt','GarageArea','GarageCars',
    'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
    'BsmtFullBath','BsmtHalfBath','MasVnrArea'
]

@st.cache_resource
def load_model():
    if os.path.exists('model/final_model.pkl'):
        model = joblib.load('model/final_model.pkl')
        with open('model/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, feature_names
    return None, None

@st.cache_data
def load_train():
    if os.path.exists('train.csv'):
        return pd.read_csv('train.csv')
    return None

def preprocess(df, feature_names, train_ref=None):
    df = df.copy()

    # None doldur
    for col in NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # Sıfır doldur
    for col in ZERO_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # LotFrontage
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = pd.to_numeric(df['LotFrontage'], errors='coerce').fillna(70)

    # Kalan eksikler
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'None')
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Feature Engineering
    df['TotalSF']          = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    df['TotalBath']        = (df.get('FullBath', 0) + 0.5*df.get('HalfBath', 0) +
                              df.get('BsmtFullBath', 0) + 0.5*df.get('BsmtHalfBath', 0))
    df['TotalPorch']       = (df.get('OpenPorchSF', 0) + df.get('EnclosedPorch', 0) +
                              df.get('3SsnPorch', 0) + df.get('ScreenPorch', 0) + df.get('WoodDeckSF', 0))
    df['HouseAge']         = df.get('YrSold', 2008) - df.get('YearBuilt', 2000)
    df['YearsSinceRemod']  = df.get('YrSold', 2008) - df.get('YearRemodAdd', 2000)
    df['IsRemodeled']      = (df.get('YearBuilt', 2000) != df.get('YearRemodAdd', 2000)).astype(int)
    df['HasGarage']        = (df.get('GarageArea', 0) > 0).astype(int)
    df['HasBsmt']          = (df.get('TotalBsmtSF', 0) > 0).astype(int)
    df['HasPool']          = (df.get('PoolArea', 0) > 0).astype(int)
    df['QualArea']         = df.get('OverallQual', 5) * df.get('GrLivArea', 1500)

    # Ordinal encoding
    for col, mapping in ORDINAL_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # One-hot encoding
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Eksik sütunları ekle, fazla sütunları çıkar
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/House_icon_-_Noun_Project_27092.svg/120px-House_icon_-_Noun_Project_27092.svg.png", width=80)
st.sidebar.title("🏠 Ev Fiyat Tahmini")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Sayfa Seçin",
    ["🏠 Ana Sayfa", "📊 Veri Keşfi (EDA)", "🎯 Tahmin Yap", "📈 Model Bilgisi"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Proje Bilgisi**
- Veri: Kaggle House Prices
- Model: LightGBM / XGBoost
- Metrik: RMSE (log ölçek)
- Kaynak: MIT Lisansı
""")

# ─── Model Yükle ──────────────────────────────────────────────────────────────
model, feature_names = load_model()
train_df = load_train()

# ══════════════════════════════════════════════════════════════════════════════
# SAYFA 1 — Ana Sayfa
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Ana Sayfa":
    st.title("🏠 Ev Fiyat Tahmin Uygulaması")
    st.markdown("""
    **Proje:** Ames, Iowa konutlarının satış fiyatını makine öğrenmesi ile tahmin etme.

    **Kaynak:** Kaggle — [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
    **Lisans:** MIT
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Eğitim Gözlemi", "1.460")
    with col2:
        st.metric("🔢 Özellik Sayısı", "79")
    with col3:
        st.metric("🎯 Hedef Değişken", "SalePrice")
    with col4:
        st.metric("📏 Ana Metrik", "RMSE")

    st.markdown("---")
    st.subheader("Proje Akışı")
    akis = {
        "Aşama": [
            "1. Problem Tanımı", "2. Veri Yükleme", "3. EDA",
            "4. Veri Kalitesi", "5. Preprocessing", "6. Veri Bölme",
            "7. Baseline", "8. Model Karşılaştırma", "9. Aday Seçim",
            "10. Hiperparametre Opt.", "11. Rakip Karşılaştırma",
            "12. Final Değerlendirme", "13. Submission"
        ],
        "Durum": ["✅"] * 13
    }
    st.dataframe(pd.DataFrame(akis), use_container_width=True, hide_index=True)

    if model is None:
        st.warning("⚠️ Model bulunamadı. Lütfen önce `ev_fiyat_tahmin.ipynb` notebook'unu çalıştırın.")
    else:
        st.success("✅ Model yüklendi. Sol menüden 'Tahmin Yap' sayfasına gidin.")

# ══════════════════════════════════════════════════════════════════════════════
# SAYFA 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Veri Keşfi (EDA)":
    st.title("📊 Veri Keşfi (EDA)")

    if train_df is None:
        st.error("❌ train.csv bulunamadı. Uygulama ile aynı klasörde olmalı.")
        st.stop()

    # Özet
    st.subheader("Veri Seti Özeti")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Satır Sayısı", f"{train_df.shape[0]:,}")
    with col2:
        st.metric("Sütun Sayısı", f"{train_df.shape[1]}")
    with col3:
        st.metric("Eksik Değerli Sütun", f"{(train_df.isnull().sum() > 0).sum()}")

    st.dataframe(train_df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("SalePrice Dağılımı")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(train_df['SalePrice'], bins=50, color='steelblue', edgecolor='white')
    axes[0].set_title('Orijinal Dağılım')
    axes[0].set_xlabel('SalePrice ($)')
    axes[1].hist(np.log1p(train_df['SalePrice']), bins=50, color='darkorange', edgecolor='white')
    axes[1].set_title('log(SalePrice) Dağılımı')
    axes[1].set_xlabel('log(SalePrice)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Sayısal Değişkenler — SalePrice Korelasyonu")
    num_cols = train_df.select_dtypes(include=[np.number]).columns.drop(['Id', 'SalePrice'])
    corr = train_df[num_cols].corrwith(train_df['SalePrice']).abs().sort_values(ascending=False)
    top_n = st.slider("Kaç özellik gösterilsin?", 5, 20, 15)
    fig2, ax2 = plt.subplots(figsize=(10, top_n * 0.4 + 1))
    corr.head(top_n).plot(kind='barh', ax=ax2, color='steelblue', edgecolor='white')
    ax2.set_xlabel('|Korelasyon|')
    ax2.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.subheader("Eksik Değer Analizi")
    missing = train_df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_df = pd.DataFrame({
        'Eksik Sayı': missing,
        'Eksik %': (missing / len(train_df) * 100).round(2)
    })
    st.dataframe(missing_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Kategorik Değişken İnceleme")
    cat_col = st.selectbox(
        "Kategorik değişken seçin:",
        ['Neighborhood', 'OverallQual', 'BldgType', 'HouseStyle', 'KitchenQual', 'GarageType']
    )
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    order = train_df.groupby(cat_col)['SalePrice'].median().sort_values(ascending=False).index
    train_df.boxplot(column='SalePrice', by=cat_col, ax=ax3, order=order, rot=45, grid=False)
    ax3.set_title(f'{cat_col} Bazlı SalePrice Dağılımı')
    ax3.set_xlabel(cat_col)
    ax3.set_ylabel('SalePrice ($)')
    plt.suptitle('')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SAYFA 3 — Tahmin Yap
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🎯 Tahmin Yap":
    st.title("🎯 Ev Fiyatı Tahmin Et")

    if model is None:
        st.error("❌ Model bulunamadı. Lütfen önce notebook'u çalıştırın (`ev_fiyat_tahmin.ipynb`).")
        st.stop()

    tab1, tab2 = st.tabs(["✍️ Manuel Giriş", "📂 CSV Yükle"])

    # ── TAB 1: Manuel Giriş ───────────────────────────────────────────────────
    with tab1:
        st.markdown("En önemli özellikler için değerleri girin:")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Temel Bilgiler**")
            overall_qual  = st.slider("Genel Kalite (1-10)", 1, 10, 7)
            gr_liv_area   = st.number_input("Yaşam Alanı (ft²)", 300, 6000, 1500, step=50)
            year_built    = st.number_input("İnşaat Yılı", 1870, 2010, 2000)
            year_remod    = st.number_input("Yenileme Yılı", 1870, 2010, 2000)
            neighborhood  = st.selectbox("Mahalle", [
                'CollgCr','Veenker','Crawfor','NoRidge','Mitchel','Somerst',
                'NWAmes','OldTown','BrkSide','Sawyer','NridgHt','NAmes',
                'SawyerW','IDOTRR','MeadowV','Edwards','Timber','Gilbert',
                'StoneBr','ClearCr','NPkVill','Blmngtn','BrDale','SWISU','Blueste'
            ], index=0)

        with col2:
            st.markdown("**Alan Bilgileri**")
            total_bsmt_sf = st.number_input("Bodrum Alanı (ft²)", 0, 3000, 800, step=50)
            first_flr_sf  = st.number_input("1. Kat Alanı (ft²)", 300, 4000, 900, step=50)
            second_flr_sf = st.number_input("2. Kat Alanı (ft²)", 0, 2000, 500, step=50)
            garage_area   = st.number_input("Garaj Alanı (ft²)", 0, 1500, 480, step=50)
            garage_cars   = st.slider("Garaj Kapasitesi (araç)", 0, 4, 2)

        with col3:
            st.markdown("**Kalite & Özellikler**")
            kitchen_qual  = st.selectbox("Mutfak Kalitesi", ['Ex','Gd','TA','Fa','Po'], index=1)
            exter_qual    = st.selectbox("Dış Cephe Kalitesi", ['Ex','Gd','TA','Fa','Po'], index=1)
            bsmt_qual     = st.selectbox("Bodrum Kalitesi", ['Ex','Gd','TA','Fa','Po','None'], index=1)
            full_bath     = st.slider("Tam Banyo Sayısı", 0, 4, 2)
            fireplaces    = st.slider("Şömine Sayısı", 0, 4, 1)
            overall_cond  = st.slider("Genel Durum (1-10)", 1, 10, 5)

        if st.button("🔮 Fiyat Tahmin Et", type="primary", use_container_width=True):
            # Girdi satırı oluştur
            input_data = {
                'MSSubClass': 60, 'MSZoning': 'RL',
                'LotFrontage': 70.0, 'LotArea': 9000,
                'Street': 'Pave', 'Alley': 'None',
                'LotShape': 'Reg', 'LandContour': 'Lvl',
                'Utilities': 'AllPub', 'LotConfig': 'Inside',
                'LandSlope': 'Gtl', 'Neighborhood': neighborhood,
                'Condition1': 'Norm', 'Condition2': 'Norm',
                'BldgType': '1Fam', 'HouseStyle': '2Story',
                'OverallQual': overall_qual, 'OverallCond': overall_cond,
                'YearBuilt': year_built, 'YearRemodAdd': year_remod,
                'RoofStyle': 'Gable', 'RoofMatl': 'CompShg',
                'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd',
                'MasVnrType': 'None', 'MasVnrArea': 0,
                'ExterQual': exter_qual, 'ExterCond': 'TA',
                'Foundation': 'PConc',
                'BsmtQual': bsmt_qual, 'BsmtCond': 'TA',
                'BsmtExposure': 'No', 'BsmtFinType1': 'Unf',
                'BsmtFinSF1': 0, 'BsmtFinType2': 'None',
                'BsmtFinSF2': 0, 'BsmtUnfSF': total_bsmt_sf,
                'TotalBsmtSF': total_bsmt_sf,
                'Heating': 'GasA', 'HeatingQC': 'Ex',
                'CentralAir': 'Y', 'Electrical': 'SBrkr',
                '1stFlrSF': first_flr_sf, '2ndFlrSF': second_flr_sf,
                'LowQualFinSF': 0, 'GrLivArea': gr_liv_area,
                'BsmtFullBath': 0, 'BsmtHalfBath': 0,
                'FullBath': full_bath, 'HalfBath': 1,
                'BedroomAbvGr': 3, 'KitchenAbvGr': 1,
                'KitchenQual': kitchen_qual, 'TotRmsAbvGrd': 8,
                'Functional': 'Typ', 'Fireplaces': fireplaces,
                'FireplaceQu': 'Gd' if fireplaces > 0 else 'None',
                'GarageType': 'Attchd', 'GarageYrBlt': year_built,
                'GarageFinish': 'RFn', 'GarageCars': garage_cars,
                'GarageArea': garage_area, 'GarageQual': 'TA',
                'GarageCond': 'TA', 'PavedDrive': 'Y',
                'WoodDeckSF': 0, 'OpenPorchSF': 60,
                'EnclosedPorch': 0, '3SsnPorch': 0,
                'ScreenPorch': 0, 'PoolArea': 0,
                'PoolQC': 'None', 'Fence': 'None',
                'MiscFeature': 'None', 'MiscVal': 0,
                'MoSold': 6, 'YrSold': 2008,
                'SaleType': 'WD', 'SaleCondition': 'Normal',
            }

            input_df = pd.DataFrame([input_data])
            try:
                processed = preprocess(input_df, feature_names)
                log_pred  = model.predict(processed)[0]
                pred_price = np.expm1(log_pred)

                st.markdown("---")
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("💰 Tahmin Edilen Fiyat", f"${pred_price:,.0f}")
                with col_res2:
                    st.metric("📉 Alt Tahmin (%10)", f"${pred_price * 0.90:,.0f}")
                with col_res3:
                    st.metric("📈 Üst Tahmin (%10)", f"${pred_price * 1.10:,.0f}")

                # Gauge chart
                fig_g, ax_g = plt.subplots(figsize=(8, 3))
                price_range = np.linspace(50000, 800000, 100)
                ax_g.barh(0, 800000, color='#f0f0f0', height=0.5)
                ax_g.barh(0, pred_price, color='steelblue', height=0.5)
                ax_g.axvline(pred_price, color='darkblue', lw=2)
                ax_g.set_xlim(0, 800000)
                ax_g.set_yticks([])
                ax_g.set_xlabel('Fiyat ($)')
                ax_g.set_title(f'Tahmin: ${pred_price:,.0f}', fontsize=14, fontweight='bold')
                ax_g.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
                plt.tight_layout()
                st.pyplot(fig_g)
                plt.close()

                st.info(f"""
                ℹ️ **Tahmin Notu:** Bu tahmin, 2006–2010 yılları arasındaki Ames, Iowa konut satış
                verilerine dayalıdır. Gerçek piyasa fiyatları bölge, dönem ve piyasa koşullarına
                göre farklılık gösterebilir. Model ±%10 hata marjıyla çalışır.
                """)
            except Exception as e:
                st.error(f"Tahmin hatası: {e}")

    # ── TAB 2: CSV Yükle ──────────────────────────────────────────────────────
    with tab2:
        st.markdown("test.csv veya kendi veri setinizi yükleyin:")
        uploaded = st.file_uploader("CSV dosyası yükleyin", type=['csv'])

        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ {len(df_up)} satır yüklendi.")
            st.dataframe(df_up.head(), use_container_width=True)

            if st.button("🔮 Toplu Tahmin Yap", type="primary"):
                try:
                    ids = df_up['Id'] if 'Id' in df_up.columns else pd.RangeIndex(len(df_up))
                    df_proc = preprocess(df_up.drop('Id', axis=1, errors='ignore'), feature_names)
                    preds = np.expm1(model.predict(df_proc))
                    result_df = pd.DataFrame({'Id': ids, 'SalePrice_Tahmin': preds.round(0)})
                    st.dataframe(result_df, use_container_width=True)

                    csv_out = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Tahminleri İndir (CSV)",
                        csv_out, "tahminler.csv", "text/csv"
                    )
                except Exception as e:
                    st.error(f"Hata: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SAYFA 4 — Model Bilgisi
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📈 Model Bilgisi":
    st.title("📈 Model Bilgisi & Deney Sonuçları")

    st.subheader("Kullanılan Modeller")
    model_df = pd.DataFrame({
        'Model': ['Baseline (Medyan)', 'Linear Regression', 'Ridge Regression',
                  'KNN Regressor', 'Decision Tree', 'Random Forest',
                  'XGBoost', 'LightGBM (Tuned) ✅'],
        'Tür': ['Heuristic', 'Parametrik', 'Parametrik (Regularized)',
                'Mesafe Tabanlı', 'Tree-Based', 'Ensemble',
                'Gradient Boosting', 'Gradient Boosting'],
        'Durum': ['Referans', 'Karşılaştırma', 'Karşılaştırma',
                  'Karşılaştırma', 'Karşılaştırma', 'Strong Competitor',
                  'Strong Competitor', 'FİNAL MODEL']
    })
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Proje Metrikleri")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Metrik Seçimi**
        - **Ana Metrik:** RMSE (log ölçek)
          - Log dönüşümü sayesinde yüksek fiyatlı evlerin orantısız etkisi azaltılır
        - **Destekleyici:** MAE, R²
        - **Başarı Kriteri:** Baseline RMSE'yi ≥%30 iyileştirmek
        """)
    with col2:
        st.markdown("""
        **Veri İşleme Kararları**
        - GrLivArea>4000 & SalePrice<300k → 2 outlier silindi
        - LotFrontage → mahalle medyanı ile dolduruldu
        - 'NA' değerleri → 'None' (özellik yok anlamında)
        - 10 yeni özellik türetildi (TotalSF, QualArea vb.)
        - Ordinal + One-Hot encoding uygulandı
        """)

    st.markdown("---")
    st.subheader("Önemli Özellikler")
    if model is not None and hasattr(model, 'feature_importances_') and feature_names is not None:
        fi = pd.Series(model.feature_importances_, index=feature_names)
        fi_top = fi.sort_values(ascending=False).head(15)
        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        fi_top.plot(kind='barh', ax=ax_fi, color='steelblue', edgecolor='white')
        ax_fi.set_title('Feature Importance — Top 15', fontsize=13, fontweight='bold')
        ax_fi.set_xlabel('Önem Skoru')
        ax_fi.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()
    else:
        st.info("Model yüklendikten sonra feature importance grafiği burada görünecek.")

    st.markdown("---")
    st.subheader("Model Sınırlılıkları")
    st.warning("""
    ⚠️ **Dikkat Edilmesi Gerekenler:**
    - Model 2006–2010 yılları Ames, Iowa verisiyle eğitilmiştir.
    - Farklı şehir, dönem veya piyasa koşullarında genelleme sınırlıdır.
    - $400.000 üzerindeki çok yüksek fiyatlı evlerde hata payı artar.
    - Tahminler yatırım kararı için tek başına kullanılmamalıdır.
    """)
