import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os
from scipy.spatial import cKDTree
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="مستشار العقار السعودي الذكي", page_icon="🏠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
html,body,[class*="css"]{font-family:'Tajawal',sans-serif;direction:rtl;}
h1,h2,h3,.stMarkdown{text-align:right;}
.result-box{background:linear-gradient(135deg,#1a4e8f,#0d7c5b);border-radius:16px;padding:28px 32px;text-align:center;margin:20px 0;color:white;}
.result-price{font-size:40px;font-weight:700;margin:0;}
.result-label{font-size:14px;opacity:.85;margin:0 0 6px;}
.result-range{font-size:14px;opacity:.75;margin:6px 0 0;}
.mcard{border-radius:10px;padding:12px 16px;margin:7px 0;border-right:4px solid #1a4e8f;background:#f0f4fd;}
.green{border-right-color:#0d7c5b;background:#f0faf5;}
.red{border-right-color:#c0392b;background:#fdf0f0;}
.amber{border-right-color:#d68910;background:#fef9f0;}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500;}
.bg{background:#d4edda;color:#155724;}
.bb{background:#d1ecf1;color:#0c5460;}
.ba{background:#fff3cd;color:#856404;}
.irow{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #eee;font-size:13px;}
.ctable{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;}
.ctable th{background:#f5f7fb;padding:7px;text-align:right;color:#555;font-weight:500;}
.ctable td{padding:7px;border-bottom:1px solid #f0f0f0;}
footer{text-align:center;color:#999;font-size:11px;margin-top:36px;}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    m={}
    for lt in ['sale','rent']:
        p=os.path.join(BASE,f'xgb_{lt}_full.pkl')
        if os.path.exists(p):
            with open(p,'rb') as f: m[lt]=pickle.load(f)
    return m

@st.cache_data
def load_hierarchy():
    with open(os.path.join(BASE,'hierarchy.json'),encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_stats():
    return pd.read_csv(os.path.join(BASE,'stats_hierarchy.csv'))

@st.cache_data
def load_posts_geo():
    return pd.read_parquet(os.path.join(BASE,'posts_geo.parquet'))

@st.cache_data
def load_centroids():
    df = pd.read_csv(os.path.join(BASE,'centroids.csv'))
    tree = cKDTree(df[['lat','lng']].values)
    return df, tree

models   = load_models()
hier     = load_hierarchy()
stats    = load_stats()
posts_geo= load_posts_geo()
cent_df, cent_tree = load_centroids()

# prebuilt KDTree for posts
@st.cache_resource
def build_posts_tree():
    coords = posts_geo[['lat','lng']].values
    return cKDTree(coords)
posts_tree = build_posts_tree()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏠 مستشار العقار السعودي الذكي")
st.markdown("تقييم فوري مبني على أسعار العقارات **القريبة فعلاً** من موقعك.")
c1,c2,c3 = st.columns(3)
with c1: st.markdown('<span class="badge bb">127,000+ إعلان</span>',unsafe_allow_html=True)
with c2: st.markdown('<span class="badge bg">R² بيع = 0.86</span>',unsafe_allow_html=True)
with c3: st.markdown('<span class="badge ba">العنوان الوطني</span>',unsafe_allow_html=True)
st.divider()

# ── Operation type ────────────────────────────────────────────────────────────
ltype_label = st.radio("نوع العملية", ["بيع","إيجار"], horizontal=True)
ltype = 'sale' if ltype_label=='بيع' else 'rent'
st.markdown("---")

# ── Location ─────────────────────────────────────────────────────────────────
st.markdown("### 📍 حدد موقع العقار")
loc_tab1, loc_tab2 = st.tabs(["🗺️ انقر على الخريطة", "📋 اختر يدوياً"])

sel_lat, sel_lng = None, None
sel_region, sel_city, sel_district = None, None, None
auto_mode = False

with loc_tab1:
    st.caption("انقر على موقع العقار في الخريطة — سيتم تحديد المنطقة والمدينة والحي تلقائياً")

    # Default center: Riyadh
    init_lat = st.session_state.get('map_lat', 24.7136)
    init_lng = st.session_state.get('map_lng', 46.6753)

    m = folium.Map(location=[init_lat, init_lng], zoom_start=11,
                   tiles='CartoDB positron')

    # Add centroids as light markers
    for _, row in cent_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=3, color='#2d5fa6', fill=True,
            fill_opacity=0.4, weight=1,
            tooltip=f"{row['city_ar']} — {row['district_ar']}"
        ).add_to(m)

    map_data = st_folium(m, height=400, width=None, returned_objects=["last_clicked"])

    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        sel_lat = clicked["lat"]
        sel_lng = clicked["lng"]
        st.session_state['map_lat'] = sel_lat
        st.session_state['map_lng'] = sel_lng

        # Find nearest district
        dist_deg, idx = cent_tree.query([[sel_lat, sel_lng]], k=1)
        dist_km = dist_deg[0][0] * 111
        row = cent_df.iloc[idx[0][0]]
        sel_region   = row['region_ar']
        sel_city     = row['city_ar']
        sel_district = row['district_ar']
        auto_mode    = True

        color_cls = "green" if dist_km <= 5 else "amber"
        st.markdown(f"""
        <div class="mcard {color_cls}">
            ✅ <strong>تم التحديد:</strong>
            &nbsp;{sel_region} ← {sel_city} ← حي {sel_district}
            &nbsp;|&nbsp; المسافة: {dist_km:.1f} كم
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("انقر على الخريطة لتحديد الموقع")

with loc_tab2:
    regions = sorted(hier.keys())
    sel_region_m = st.selectbox("🗺️ المنطقة", regions,
                                 index=regions.index('منطقة الرياض') if 'منطقة الرياض' in regions else 0)
    cities_in_region = sorted(hier[sel_region_m].keys())
    sel_city_m = st.selectbox("🏙️ المدينة",  cities_in_region,
                               index=cities_in_region.index('الرياض') if 'الرياض' in cities_in_region else 0)
    districts_in_city = hier[sel_region_m][sel_city_m]
    sel_district_m = st.selectbox("🏘️ الحي", districts_in_city)

    # Get centroid for manual selection
    row_m = cent_df[(cent_df['city_ar']==sel_city_m) &
                    (cent_df['district_ar']==sel_district_m)]
    if not row_m.empty:
        sel_lat      = row_m.iloc[0]['lat']
        sel_lng      = row_m.iloc[0]['lng']
        sel_region   = sel_region_m
        sel_city     = sel_city_m
        sel_district = sel_district_m

st.markdown("---")

# ── Property details ──────────────────────────────────────────────────────────
st.markdown("### 🏗️ تفاصيل العقار")
d1,d2,d3 = st.columns(3)
with d1: area  = st.number_input("📐 المساحة (م²)", min_value=20, max_value=50000, value=150, step=10)
with d2: rooms = st.number_input("🛏️ عدد الغرف",    min_value=1, max_value=12,    value=3,   step=1)
with d3: baths = st.number_input("🚿 الحمامات",      min_value=1, max_value=10,    value=2,   step=1)

a1,a2,a3,a4 = st.columns(4)
with a1: has_parking  = st.checkbox("🚗 موقف",  value=True)
with a2: has_elevator = st.checkbox("🛗 مصعد",  value=False)
with a3: has_ac       = st.checkbox("❄️ مكيف",  value=False)
with a4: is_furnished = st.checkbox("🛋️ مفروش", value=False)

radius_km = st.slider("🔍 نطاق البحث عن العقارات القريبة (كم)", 1, 10, 3)
st.divider()

predict_btn = st.button("🔍 تقييم العقار", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not sel_lat or not sel_city:
        st.error("الرجاء تحديد الموقع أولاً — انقر على الخريطة أو اختر يدوياً.")
        st.stop()

    if ltype not in models:
        st.error("النموذج غير متوفر.")
        st.stop()

    # ── 1. ML Model prediction ────────────────────────────────────────────────
    pkg     = models[ltype]
    model   = pkg['model']
    le_city = pkg['le_city']
    le_nbhd = pkg['le_nbhd']
    city_enc = le_city.transform([sel_city])[0]     if sel_city     in le_city.classes_ else 0
    nbhd_enc = le_nbhd.transform([sel_district])[0] if sel_district in le_nbhd.classes_ else 0

    X_in = pd.DataFrame([{
        'city_enc':city_enc,'nbhd_enc':nbhd_enc,
        'area_clean':area,'rooms_clean':rooms,'baths_clean':baths,
        'has_elevator':int(has_elevator),'has_parking':int(has_parking),
        'has_ac':int(has_ac),'is_furnished':int(is_furnished),
    }])
    ml_ppsqm = np.expm1(model.predict(X_in)[0])

    # ── 2. Nearby properties actual prices ────────────────────────────────────
    radius_deg = radius_km / 111.0
    nearby_idx = posts_tree.query_ball_point([sel_lat, sel_lng], r=radius_deg)
    nearby = posts_geo.iloc[nearby_idx]
    nearby_ltype = nearby[nearby['listing_type_key']==ltype]

    if len(nearby_ltype) >= 5:
        nearby_median = nearby_ltype['price_per_sqm'].median()
        nearby_mean   = nearby_ltype['price_per_sqm'].mean()
        # Blend: 60% ML + 40% nearby median
        blended_ppsqm = ml_ppsqm * 0.60 + nearby_median * 0.40
        use_nearby = True
    else:
        blended_ppsqm = ml_ppsqm
        use_nearby = False

    pred_total = blended_ppsqm * area
    low_total  = blended_ppsqm * 0.85 * area
    high_total = blended_ppsqm * 1.15 * area

    nearby_note = f"مدمج مع {len(nearby_ltype)} عقار قريب ({radius_km}كم)" if use_nearby else "نموذج AI فقط (لا توجد بيانات قريبة كافية)"

    # ── Result box ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-box">
        <p class="result-label">
            {'🏠 سعر المتر المربع' if ltype=='sale' else '🔑 الإيجار السنوي/م²'}
            &nbsp;·&nbsp; {sel_region} ← {sel_city} ← حي {sel_district}
        </p>
        <p class="result-price">{blended_ppsqm:,.0f} ريال/م²</p>
        <p class="result-range">
            إجمالي ({area:,} م²): <strong>{pred_total:,.0f} ريال</strong>
        </p>
        <p class="result-range">النطاق: {low_total:,.0f} — {high_total:,.0f} ريال</p>
        <p class="result-range" style="font-size:12px;margin-top:6px;opacity:.65;">
            🔬 {nearby_note}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Nearby properties analysis ────────────────────────────────────────────
    if use_nearby:
        st.markdown("#### 🏘️ العقارات القريبة منك")
        n_sale = len(nearby[nearby['listing_type_key']=='sale'])
        n_rent = len(nearby[nearby['listing_type_key']=='rent'])
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("إعلانات للبيع قريبة",  f"{n_sale:,}")
        col_b.metric("إعلانات للإيجار قريبة", f"{n_rent:,}")
        col_c.metric("وسيط سعر/م² قريب", f"{nearby_median:,.0f}")

        # Map with nearby properties
        st.markdown("**خريطة العقارات القريبة:**")
        m2 = folium.Map(location=[sel_lat, sel_lng], zoom_start=14, tiles='CartoDB positron')

        # User location
        folium.Marker(
            [sel_lat, sel_lng],
            popup="📍 موقع العقار",
            icon=folium.Icon(color='red', icon='home')
        ).add_to(m2)

        # Nearby listings
        for _, row in nearby_ltype.head(50).iterrows():
            color = 'blue' if row['listing_type_key']=='sale' else 'green'
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=5, color=color, fill=True, fill_opacity=0.6,
                tooltip=f"{row['price_per_sqm']:,.0f} ريال/م²"
            ).add_to(m2)

        # Search radius circle
        folium.Circle(
            [sel_lat, sel_lng],
            radius=radius_km*1000,
            color='#2d5fa6', fill=True, fill_opacity=0.05, weight=1.5
        ).add_to(m2)

        st_folium(m2, height=300, width=None, returned_objects=[])

    # ── District stats comparison ─────────────────────────────────────────────
    st.markdown("#### 📊 مقارنة بالسوق")
    dist_row  = stats[(stats['city']==sel_city) & (stats['district']==sel_district) & (stats['ltype']==ltype)]
    city_rows = stats[(stats['city']==sel_city) & (stats['ltype']==ltype)]

    if not dist_row.empty:
        mkt  = dist_row.iloc[0]['median_ppsqm']
        cnt  = int(dist_row.iloc[0]['count'])
        diff = (blended_ppsqm - mkt) / mkt * 100
        if diff > 15:
            cls, icon, msg = 'red',   '⚠️', f"التقدير أعلى من وسيط الحي بـ {diff:.0f}%"
        elif diff < -15:
            cls, icon, msg = 'green', '✅', f"التقدير أقل من وسيط الحي بـ {abs(diff):.0f}% — سعر مناسب"
        else:
            cls, icon, msg = 'mcard', '📊', f"التقدير ضمن النطاق الطبيعي للحي"
        st.markdown(f"""
        <div class="mcard {cls}">
            <strong>{icon} {msg}</strong><br>
            <small>وسيط حي {sel_district}: {mkt:,.0f} ريال/م² · {cnt} إعلان</small>
        </div>
        """, unsafe_allow_html=True)

    if not city_rows.empty:
        city_med = city_rows['median_ppsqm'].median()
        city_cnt = int(city_rows['count'].sum())
        st.markdown(f"""
        <div class="mcard amber">
            📍 <strong>وسيط مدينة {sel_city}:</strong> {city_med:,.0f} ريال/م²
            <small>· {city_cnt:,} إعلان</small>
        </div>
        """, unsafe_allow_html=True)

    # ── Top active neighborhoods ──────────────────────────────────────────────
    top = city_rows.nlargest(5,'count')[['district','median_ppsqm','count']]
    if not top.empty:
        st.markdown(f"#### 🏘️ أكثر أحياء {sel_city} نشاطاً")
        rows_html = "".join(
            f"<tr><td>{'★ ' if r['district']==sel_district else ''}{r['district']}</td>"
            f"<td>{r['median_ppsqm']:,.0f}</td><td>{int(r['count']):,}</td></tr>"
            for _,r in top.iterrows()
        )
        st.markdown(f"""
        <table class="ctable">
          <thead><tr><th>الحي</th><th>وسيط/م²</th><th>إعلانات</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

    # ── Feature impact ────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ تأثير المواصفات")
    for feat,label in [('has_parking','🚗 موقف'),('has_elevator','🛗 مصعد'),
                        ('is_furnished','🛋️ مفروش'),('has_ac','❄️ مكيف')]:
        ron  = X_in.copy(); ron[feat]  = 1
        roff = X_in.copy(); roff[feat] = 0
        d    = np.expm1(model.predict(ron)[0]) - np.expm1(model.predict(roff)[0])
        sign = "+" if d>=0 else ""
        color= "#155724" if d>=0 else "#721c24"
        st.markdown(f'<div class="irow"><span>{label}</span>'
                    f'<span style="color:{color};font-weight:500;">{sign}{d:,.0f} ريال/م²</span></div>',
                    unsafe_allow_html=True)

    st.markdown("""
    <footer>
        بيانات: عقار · بيوت · وصلت · PropertyFinder · العنوان الوطني السعودي<br>
        للأغراض التوجيهية فقط · لا يُعتمد عليه للقرارات المالية النهائية
    </footer>""", unsafe_allow_html=True)
