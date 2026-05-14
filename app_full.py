import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

st.set_page_config(
    page_title="مستشار العقار السعودي الذكي",
    page_icon="🏠",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family:'Tajawal',sans-serif; direction:rtl; }
h1,h2,h3,.stMarkdown { text-align:right; }

.result-box {
    background: linear-gradient(135deg,#1a4e8f,#0d7c5b);
    border-radius:16px; padding:28px 32px;
    text-align:center; margin:20px 0; color:white;
}
.result-price { font-size:42px; font-weight:700; margin:0; }
.result-label { font-size:15px; opacity:.85; margin:0 0 8px; }
.result-range { font-size:15px; opacity:.75; margin:8px 0 0; }

.market-card {
    border-radius:12px; padding:14px 18px; margin:8px 0;
    border-right:4px solid #1a4e8f;
    background:#f0f4fd;
}
.verdict-green { border-right-color:#0d7c5b; background:#f0faf5; }
.verdict-red   { border-right-color:#c0392b; background:#fdf0f0; }
.verdict-blue  { border-right-color:#1a4e8f; background:#f0f4fd; }
.verdict-amber { border-right-color:#d68910; background:#fef9f0; }

.badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:500; }
.badge-green { background:#d4edda; color:#155724; }
.badge-blue  { background:#d1ecf1; color:#0c5460; }
.badge-amber { background:#fff3cd; color:#856404; }

.insight-row {
    display:flex; justify-content:space-between;
    padding:8px 0; border-bottom:1px solid #eee; font-size:14px;
}
.comp-table { width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }
.comp-table th { background:#f5f7fb; padding:8px; text-align:right; color:#555; font-weight:500; }
.comp-table td { padding:8px; border-bottom:1px solid #f0f0f0; }
footer { text-align:center; color:#999; font-size:12px; margin-top:40px; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    models = {}
    for ltype in ['sale','rent']:
        path = os.path.join(BASE, f'xgb_{ltype}_full.pkl')
        if os.path.exists(path):
            with open(path,'rb') as f:
                models[ltype] = pickle.load(f)
    return models

@st.cache_data
def load_mapping():
    with open(os.path.join(BASE,'city_neighborhoods_full.json'), encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_stats():
    return pd.read_csv(os.path.join(BASE,'neighborhood_stats_full.csv'))

models  = load_models()
mapping = load_mapping()
stats   = load_stats()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏠 مستشار العقار السعودي الذكي")
st.markdown("أدخل تفاصيل العقار للحصول على تقييم فوري ومقارنة بالسوق.")

cols = st.columns(3)
with cols[0]: st.markdown('<span class="badge badge-blue">127,000+ إعلان</span>', unsafe_allow_html=True)
with cols[1]: st.markdown('<span class="badge badge-green">R² بيع = 0.86</span>', unsafe_allow_html=True)
with cols[2]: st.markdown('<span class="badge badge-amber">R² إيجار = 0.60</span>', unsafe_allow_html=True)
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
ltype_label = st.radio("نوع العملية", ["بيع", "إيجار"], horizontal=True)
ltype = 'sale' if ltype_label == 'بيع' else 'rent'

st.markdown("---")

# Row 1: City + Neighborhood
col1, col2 = st.columns(2)
with col1:
    city_list = sorted([c for c in mapping.keys() if c not in ('___','---')])
    default_city = city_list.index('الرياض') if 'الرياض' in city_list else 0
    city = st.selectbox("🏙️ المدينة", city_list, index=default_city)
with col2:
    nbhds = sorted([n for n in mapping.get(city,[]) if n not in ('___','---','')])
    if not nbhds: nbhds = ['غير محدد']
    neighborhood = st.selectbox("📍 الحي", nbhds)

# Row 2: Area + Rooms + Baths
col3, col4, col5 = st.columns(3)
with col3:
    area  = st.number_input("📐 المساحة (م²)", min_value=20, max_value=50000, value=150, step=10)
with col4:
    rooms = st.number_input("🛏️ عدد الغرف", min_value=1, max_value=12, value=3, step=1)
with col5:
    baths = st.number_input("🚿 عدد الحمامات", min_value=1, max_value=10, value=2, step=1)

# Row 3: Amenities
st.markdown("**المرافق والمميزات:**")
col6, col7, col8, col9 = st.columns(4)
with col6:
    has_parking  = st.checkbox("🚗 موقف سيارة", value=True)
with col7:
    has_elevator = st.checkbox("🛗 مصعد", value=False)
with col8:
    has_ac       = st.checkbox("❄️ مكيف مركزي", value=False)
with col9:
    is_furnished = st.checkbox("🛋️ مفروش", value=False)

predict_btn = st.button("🔍 تقييم العقار", use_container_width=True, type="primary")

# ── Predict ───────────────────────────────────────────────────────────────────
if predict_btn:
    if ltype not in models:
        st.error("النموذج غير متوفر، تحقق من الملفات.")
        st.stop()

    pkg     = models[ltype]
    model   = pkg['model']
    le_city = pkg['le_city']
    le_nbhd = pkg['le_nbhd']

    city_enc = le_city.transform([city])[0] if city in le_city.classes_ else 0
    nbhd_enc = le_nbhd.transform([neighborhood])[0] if neighborhood in le_nbhd.classes_ else 0

    X_in = pd.DataFrame([{
        'city_enc':   city_enc, 'nbhd_enc':   nbhd_enc,
        'area_clean': area,     'rooms_clean': rooms,
        'baths_clean':baths,    'has_elevator':int(has_elevator),
        'has_parking':int(has_parking), 'has_ac':int(has_ac),
        'is_furnished':int(is_furnished),
    }])

    pred_ppsqm = np.expm1(model.predict(X_in)[0])
    pred_total = pred_ppsqm * area
    low_total  = pred_ppsqm * 0.85 * area
    high_total = pred_ppsqm * 1.15 * area

    unit = "ريال/م²" if ltype == 'sale' else "ريال/م²"

    st.markdown(f"""
    <div class="result-box">
        <p class="result-label">سعر المتر المربع المتوقع — {'بيع' if ltype=='sale' else 'إيجار'}</p>
        <p class="result-price">{pred_ppsqm:,.0f} ريال/م²</p>
        <p class="result-range">
            {'إجمالي العقار' if ltype=='sale' else 'الإيجار السنوي المتوقع'}
            ({area:,} م²): <strong>{pred_total:,.0f} ريال</strong>
        </p>
        <p class="result-range">النطاق: {low_total:,.0f} — {high_total:,.0f} ريال</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Market comparison ─────────────────────────────────────────────────────
    st.markdown("#### 📊 مقارنة بالسوق")

    nbhd_row = stats[(stats['city']==city) &
                     (stats['neighborhood']==neighborhood) &
                     (stats['ltype']==ltype)]
    city_rows = stats[(stats['city']==city) & (stats['ltype']==ltype)]

    if not nbhd_row.empty:
        mkt = nbhd_row.iloc[0]['median_ppsqm']
        cnt = int(nbhd_row.iloc[0]['count'])
        diff = (pred_ppsqm - mkt) / mkt * 100
        if diff > 15:
            cls, icon = 'verdict-red', '⚠️'
            msg = f"التقدير أعلى من وسيط الحي بـ {diff:.0f}%"
        elif diff < -15:
            cls, icon = 'verdict-green', '✅'
            msg = f"التقدير أقل من وسيط الحي بـ {abs(diff):.0f}% — سعر مناسب"
        else:
            cls, icon = 'verdict-blue', '📊'
            msg = f"التقدير ضمن النطاق الطبيعي لحي {neighborhood}"

        st.markdown(f"""
        <div class="market-card {cls}">
            <strong>{icon} {msg}</strong><br>
            <small>وسيط الحي: {mkt:,.0f} ريال/م² · مبني على {cnt} إعلان</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"لا توجد بيانات كافية لحي {neighborhood} في هذا النوع.")

    if not city_rows.empty:
        city_med = city_rows['median_ppsqm'].median()
        city_cnt = int(city_rows['count'].sum())
        st.markdown(f"""
        <div class="market-card verdict-amber">
            📍 <strong>وسيط مدينة {city}:</strong> {city_med:,.0f} ريال/م²
            <small> · {city_cnt:,} إعلان</small>
        </div>
        """, unsafe_allow_html=True)

    # ── Top neighborhoods in city ─────────────────────────────────────────────
    top_nbhds = city_rows.nlargest(5, 'count')[['neighborhood','median_ppsqm','count']]
    if not top_nbhds.empty:
        st.markdown("#### 🏘️ أعلى الأحياء نشاطاً في " + city)
        rows_html = ""
        for _, r in top_nbhds.iterrows():
            rows_html += f"<tr><td>{r['neighborhood']}</td><td>{r['median_ppsqm']:,.0f}</td><td>{int(r['count']):,}</td></tr>"
        st.markdown(f"""
        <table class="comp-table">
          <thead><tr><th>الحي</th><th>وسيط السعر/م²</th><th>عدد الإعلانات</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    # ── Feature impact ────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ تأثير المواصفات على السعر")
    for feat, label in [('has_parking','موقف سيارة'),('has_elevator','مصعد'),
                        ('is_furnished','مفروش'),('has_ac','مكيف مركزي')]:
        r_on  = X_in.copy(); r_on[feat]  = 1
        r_off = X_in.copy(); r_off[feat] = 0
        diff  = np.expm1(model.predict(r_on)[0]) - np.expm1(model.predict(r_off)[0])
        sign  = "+" if diff >= 0 else ""
        color = "#155724" if diff >= 0 else "#721c24"
        st.markdown(f"""
        <div class="insight-row">
            <span>{label}</span>
            <span style="color:{color};font-weight:500;">{sign}{diff:,.0f} ريال/م²</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <footer>
        نموذج مبني على 127,000+ إعلان من عقار · بيوت · وصلت · PropertyFinder<br>
        للأغراض التوجيهية فقط · لا يُعتمد عليه للقرارات المالية النهائية
    </footer>
    """, unsafe_allow_html=True)
