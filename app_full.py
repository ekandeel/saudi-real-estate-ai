import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os
from scipy.spatial import cKDTree
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="مستشار العقار السعودي الذكي",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
html,body,[class*="css"]{font-family:'Tajawal',sans-serif;direction:rtl;}
h1,h2,h3,.stMarkdown{text-align:right;}
.result-box{
    background:linear-gradient(135deg,#1a4e8f,#0d7c5b);
    border-radius:16px;padding:24px 28px;text-align:center;margin:16px 0;color:white;
}
.result-price{font-size:36px;font-weight:700;margin:0;}
.result-label{font-size:13px;opacity:.85;margin:0 0 6px;}
.result-range{font-size:13px;opacity:.75;margin:5px 0 0;}
.mcard{border-radius:10px;padding:11px 14px;margin:6px 0;border-right:4px solid #1a4e8f;background:#f0f4fd;}
.green{border-right-color:#0d7c5b;background:#f0faf5;}
.red{border-right-color:#c0392b;background:#fdf0f0;}
.amber{border-right-color:#d68910;background:#fef9f0;}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;}
.bg{background:#d4edda;color:#155724;}
.bb{background:#d1ecf1;color:#0c5460;}
.ba{background:#fff3cd;color:#856404;}
.irow{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #eee;font-size:13px;}
.ctable{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px;}
.ctable th{background:#f5f7fb;padding:6px;text-align:right;color:#555;font-weight:500;}
.ctable td{padding:6px;border-bottom:1px solid #f0f0f0;}
footer{text-align:center;color:#999;font-size:11px;margin-top:24px;}
.selected-loc{background:#e8f4fd;border:1.5px solid #2d5fa6;border-radius:10px;
    padding:10px 14px;margin:8px 0;font-size:13px;}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m = {}
    for lt in ['sale','rent']:
        p = os.path.join(BASE, f'xgb_{lt}_full.pkl')
        if os.path.exists(p):
            with open(p,'rb') as f: m[lt] = pickle.load(f)
    return m

@st.cache_data
def load_hierarchy():
    with open(os.path.join(BASE,'hierarchy.json'), encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_stats():
    return pd.read_csv(os.path.join(BASE,'stats_hierarchy.csv'))

@st.cache_data
def load_posts_geo():
    return pd.read_csv(os.path.join(BASE,'posts_geo.csv'))

@st.cache_data
def load_centroids():
    return pd.read_csv(os.path.join(BASE,'centroids.csv'))

@st.cache_resource
def build_trees():
    cent = load_centroids()
    posts = load_posts_geo()
    cent_tree  = cKDTree(cent[['lat','lng']].values)
    posts_tree = cKDTree(posts[['lat','lng']].values)
    return cent_tree, posts_tree

models     = load_models()
hier       = load_hierarchy()
stats      = load_stats()
posts_geo  = load_posts_geo()
cent_df    = load_centroids()
cent_tree, posts_tree = build_trees()

# ── Init session state ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        'map_center': [24.7136, 46.6753],
        'map_zoom':   11,
        'sel_region': 'منطقة الرياض',
        'sel_city':   'الرياض',
        'sel_district': None,
        'sel_lat':    24.7136,
        'sel_lng':    46.6753,
        'source':     'dropdown',   # 'dropdown' or 'map_click'
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏠 مستشار العقار السعودي الذكي")
c1,c2,c3 = st.columns(3)
with c1: st.markdown('<span class="badge bb">127,000+ إعلان</span>', unsafe_allow_html=True)
with c2: st.markdown('<span class="badge bg">R² بيع = 0.875</span>', unsafe_allow_html=True)
with c3: st.markdown('<span class="badge ba">العنوان الوطني السعودي</span>', unsafe_allow_html=True)
st.divider()

# ── Main layout: Left panel + Map ─────────────────────────────────────────────
left_col, map_col = st.columns([1, 1.8])

# ═══════════════════════════════════════════════
# LEFT PANEL — controls
# ═══════════════════════════════════════════════
with left_col:

    # Operation type
    ltype_label = st.radio("نوع العملية", ["بيع","إيجار"], horizontal=True)
    ltype = 'sale' if ltype_label == 'بيع' else 'rent'

    st.markdown("---")
    st.markdown("**📍 اختر الموقع**")
    st.caption("اختر من القائمة أو انقر مباشرة على الخريطة")

    # Region
    regions = sorted(hier.keys())
    reg_idx = regions.index(st.session_state['sel_region']) if st.session_state['sel_region'] in regions else 0
    sel_region = st.selectbox("🗺️ المنطقة", regions, index=reg_idx, key='dd_region')

    # City
    cities = sorted(hier.get(sel_region, {}).keys())
    cur_city = st.session_state['sel_city'] if st.session_state['sel_city'] in cities else cities[0] if cities else ''
    city_idx = cities.index(cur_city) if cur_city in cities else 0
    sel_city = st.selectbox("🏙️ المدينة", cities, index=city_idx, key='dd_city')

    # District
    districts = hier.get(sel_region, {}).get(sel_city, [])
    cur_dist = st.session_state['sel_district']
    dist_idx = districts.index(cur_dist) if cur_dist in districts else 0
    sel_district = st.selectbox("🏘️ الحي", districts, index=dist_idx, key='dd_district') if districts else None

    # Sync dropdown → map center
    if sel_district and districts:
        row = cent_df[(cent_df['city_ar']==sel_city) & (cent_df['district_ar']==sel_district)]
        if not row.empty:
            new_lat = row.iloc[0]['lat']
            new_lng = row.iloc[0]['lng']
            if (abs(new_lat - st.session_state['sel_lat']) > 0.001 or
                abs(new_lng - st.session_state['sel_lng']) > 0.001):
                st.session_state['sel_lat']    = new_lat
                st.session_state['sel_lng']    = new_lng
                st.session_state['map_center'] = [new_lat, new_lng]
                st.session_state['map_zoom']   = 14
                st.session_state['source']     = 'dropdown'
    elif sel_city:
        city_rows = cent_df[cent_df['city_ar']==sel_city]
        if not city_rows.empty:
            st.session_state['map_center'] = [city_rows['lat'].mean(), city_rows['lng'].mean()]
            st.session_state['map_zoom']   = 12

    # Show current selection
    if sel_district:
        st.markdown(f"""
        <div class="selected-loc">
            📍 <strong>{sel_region}</strong><br>
            🏙️ {sel_city} ← 🏘️ حي {sel_district}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🏗️ تفاصيل العقار**")

    d1, d2 = st.columns(2)
    with d1:
        area  = st.number_input("📐 المساحة م²", min_value=20, max_value=50000, value=150, step=10)
        rooms = st.number_input("🛏️ الغرف", min_value=1, max_value=12, value=3, step=1)
    with d2:
        baths = st.number_input("🚿 الحمامات", min_value=1, max_value=10, value=2, step=1)
        radius_km = st.number_input("🔍 نطاق البحث كم", min_value=1, max_value=15, value=3, step=1)

    a1, a2 = st.columns(2)
    with a1:
        has_parking  = st.checkbox("🚗 موقف", value=True)
        has_ac       = st.checkbox("❄️ مكيف", value=False)
    with a2:
        has_elevator = st.checkbox("🛗 مصعد", value=False)
        is_furnished = st.checkbox("🛋️ مفروش", value=False)

    predict_btn = st.button("🔍 تقييم العقار", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════
# RIGHT PANEL — map
# ═══════════════════════════════════════════════
with map_col:
    st.markdown("**🗺️ الخريطة — انقر لتحديد موقع العقار**")

    sel_lat = st.session_state['sel_lat']
    sel_lng = st.session_state['sel_lng']

    # Build map
    m = folium.Map(
        location=st.session_state['map_center'],
        zoom_start=st.session_state['map_zoom'],
        tiles='CartoDB positron'
    )

    # Show district centroids for current city as light dots
    city_cents = cent_df[cent_df['city_ar'] == sel_city]
    for _, crow in city_cents.iterrows():
        is_selected = (sel_district and crow['district_ar'] == sel_district)
        folium.CircleMarker(
            location=[crow['lat'], crow['lng']],
            radius=6 if is_selected else 4,
            color='#c0392b' if is_selected else '#2d5fa6',
            fill=True,
            fill_opacity=0.85 if is_selected else 0.4,
            weight=2 if is_selected else 1,
            tooltip=f"حي {crow['district_ar']} | {int(crow['post_count'])} إعلان"
        ).add_to(m)

    # Show nearby listings (if radius set)
    radius_deg = radius_km / 111.0
    nearby_idx = posts_tree.query_ball_point([sel_lat, sel_lng], r=radius_deg)
    nearby = posts_geo.iloc[nearby_idx]
    nearby_ltype = nearby[nearby['listing_type_key'] == ltype]

    for _, prow in nearby_ltype.head(150).iterrows():
        price_label = f"{prow['price_per_sqm']:,.0f} ر/م²"
        folium.CircleMarker(
            location=[prow['lat'], prow['lng']],
            radius=4,
            color='#27ae60',
            fill=True,
            fill_opacity=0.6,
            weight=1,
            tooltip=price_label
        ).add_to(m)

    # Search radius circle
    folium.Circle(
        [sel_lat, sel_lng],
        radius=radius_km * 1000,
        color='#2d5fa6',
        fill=True,
        fill_opacity=0.04,
        weight=1.5,
        dash_array='6'
    ).add_to(m)

    # Selected location marker
    folium.Marker(
        [sel_lat, sel_lng],
        tooltip="📍 موقع العقار المختار",
        icon=folium.Icon(color='red', icon='home', prefix='fa')
    ).add_to(m)

    # Legend
    legend = f"""
    <div style="position:fixed;bottom:20px;right:20px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                border:1px solid #ddd;font-family:Tajawal,sans-serif;
                font-size:12px;direction:rtl;min-width:160px;">
        <div style="margin-bottom:5px;font-weight:600">دليل الألوان</div>
        <div>🔴 الحي المختار</div>
        <div>🔵 أحياء المدينة</div>
        <div>🟢 إعلانات قريبة ({len(nearby_ltype)})</div>
        <div style="margin-top:4px;color:#666">نطاق: {radius_km} كم</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    # Render map and capture click
    map_data = st_folium(m, height=520, use_container_width=True,
                         returned_objects=["last_clicked"])

    # Handle map click → update dropdowns
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        click_lat = clicked["lat"]
        click_lng = clicked["lng"]

        # Find nearest district
        dist_arr, idx_arr = cent_tree.query([[click_lat, click_lng]], k=1)
        dist_km_click = float(dist_arr.ravel()[0]) * 111
        nearest_row = cent_df.iloc[int(idx_arr.ravel()[0])]

        # Update session state
        st.session_state['sel_lat']      = click_lat
        st.session_state['sel_lng']      = click_lng
        st.session_state['sel_region']   = nearest_row['region_ar']
        st.session_state['sel_city']     = nearest_row['city_ar']
        st.session_state['sel_district'] = nearest_row['district_ar']
        st.session_state['map_center']   = [click_lat, click_lng]
        st.session_state['source']       = 'map_click'

        icon_c = "✅" if dist_km_click <= 5 else "⚠️"
        st.markdown(f"""
        <div class="mcard {'green' if dist_km_click<=5 else 'amber'}">
            {icon_c} نقرت على: <strong>{nearest_row['city_ar']}</strong>
            ← حي <strong>{nearest_row['district_ar']}</strong>
            &nbsp;|&nbsp; {dist_km_click:.1f} كم من مركز الحي
        </div>
        """, unsafe_allow_html=True)

        st.rerun()

# ── Prediction results (full width below) ─────────────────────────────────────
if predict_btn:
    if not sel_district:
        st.error("الرجاء اختيار حي أولاً.")
        st.stop()
    if ltype not in models:
        st.error("النموذج غير متوفر.")
        st.stop()

    pkg     = models[ltype]
    model   = pkg['model']
    le_city = pkg['le_city']
    le_nbhd = pkg['le_nbhd']

    city_enc = le_city.transform([sel_city])[0]     if sel_city     in le_city.classes_ else 0
    nbhd_enc = le_nbhd.transform([sel_district])[0] if sel_district in le_nbhd.classes_ else 0

    X_in = pd.DataFrame([{
        'city_enc':city_enc, 'nbhd_enc':nbhd_enc,
        'area_clean':area, 'rooms_clean':rooms, 'baths_clean':baths,
        'has_elevator':int(has_elevator), 'has_parking':int(has_parking),
        'has_ac':int(has_ac), 'is_furnished':int(is_furnished),
    }])

    ml_ppsqm = np.expm1(model.predict(X_in)[0])

    # Nearby blend
    radius_deg2 = radius_km / 111.0
    nb_idx  = posts_tree.query_ball_point([sel_lat, sel_lng], r=radius_deg2)
    nb_data = posts_geo.iloc[nb_idx]
    nb_type = nb_data[nb_data['listing_type_key'] == ltype]

    if len(nb_type) >= 5:
        nb_med = nb_type['price_per_sqm'].median()
        final_ppsqm = ml_ppsqm * 0.60 + nb_med * 0.40
        nearby_note = f"مدمج مع {len(nb_type)} عقار قريب ({radius_km}كم)"
    else:
        final_ppsqm = ml_ppsqm
        nb_med = None
        nearby_note = "نموذج AI فقط — لا توجد إعلانات قريبة كافية"

    pred_total = final_ppsqm * area

    # Results row
    r1, r2, r3 = st.columns([1.5, 1, 1])

    with r1:
        st.markdown(f"""
        <div class="result-box">
            <p class="result-label">
                {'🏠 سعر المتر — بيع' if ltype=='sale' else '🔑 إيجار سنوي/م²'}
                &nbsp;·&nbsp; {sel_city} ← {sel_district}
            </p>
            <p class="result-price">{final_ppsqm:,.0f} ريال/م²</p>
            <p class="result-range">
                إجمالي {area:,} م²: <strong>{pred_total:,.0f} ريال</strong>
            </p>
            <p class="result-range">
                نطاق: {final_ppsqm*0.85*area:,.0f} — {final_ppsqm*1.15*area:,.0f}
            </p>
            <p class="result-range" style="font-size:11px;opacity:.6;margin-top:4px">
                🔬 {nearby_note}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown("**مقارنة بالسوق**")
        dist_row  = stats[(stats['city']==sel_city) & (stats['district']==sel_district) & (stats['ltype']==ltype)]
        city_rows = stats[(stats['city']==sel_city) & (stats['ltype']==ltype)]

        if not dist_row.empty:
            mkt  = dist_row.iloc[0]['median_ppsqm']
            cnt  = int(dist_row.iloc[0]['count'])
            diff = (final_ppsqm - mkt) / mkt * 100
            if   diff >  15: cls,icon,msg = 'red',  '⚠️', f"أعلى من وسيط الحي بـ {diff:.0f}%"
            elif diff < -15: cls,icon,msg = 'green','✅', f"أقل من وسيط الحي بـ {abs(diff):.0f}%"
            else:             cls,icon,msg = 'mcard','📊', "ضمن النطاق الطبيعي"
            st.markdown(f"""
            <div class="mcard {cls}">
                <strong>{icon} {msg}</strong><br>
                <small>وسيط الحي: {mkt:,.0f} ر/م² · {cnt} إعلان</small>
            </div>
            """, unsafe_allow_html=True)

        if not city_rows.empty:
            city_med = city_rows['median_ppsqm'].median()
            st.markdown(f"""
            <div class="mcard amber">
                📍 وسيط {sel_city}: <strong>{city_med:,.0f}</strong> ر/م²
            </div>
            """, unsafe_allow_html=True)

        if nb_med:
            st.markdown(f"""
            <div class="mcard green">
                🟢 وسيط القريب: <strong>{nb_med:,.0f}</strong> ر/م²
                <small>({len(nb_type)} إعلان)</small>
            </div>
            """, unsafe_allow_html=True)

    with r3:
        st.markdown("**تأثير المواصفات**")
        for feat,label in [('has_parking','🚗 موقف'),('has_elevator','🛗 مصعد'),
                            ('is_furnished','🛋️ مفروش'),('has_ac','❄️ مكيف')]:
            ron  = X_in.copy(); ron[feat]  = 1
            roff = X_in.copy(); roff[feat] = 0
            d    = np.expm1(model.predict(ron)[0]) - np.expm1(model.predict(roff)[0])
            sign = "+" if d >= 0 else ""
            color= "#155724" if d >= 0 else "#721c24"
            st.markdown(
                f'<div class="irow"><span>{label}</span>'
                f'<span style="color:{color};font-weight:500;">{sign}{d:,.0f}</span></div>',
                unsafe_allow_html=True
            )

    # Top neighborhoods
    if not city_rows.empty:
        top = city_rows.nlargest(5,'count')[['district','median_ppsqm','count']]
        st.markdown(f"**🏘️ أكثر أحياء {sel_city} نشاطاً**")
        rows_html = "".join(
            f"<tr><td>{'★ ' if r['district']==sel_district else ''}{r['district']}</td>"
            f"<td>{r['median_ppsqm']:,.0f}</td><td>{int(r['count']):,}</td></tr>"
            for _, r in top.iterrows()
        )
        st.markdown(f"""
        <table class="ctable">
          <thead><tr><th>الحي</th><th>وسيط ر/م²</th><th>إعلانات</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

    st.markdown("""
    <footer>
        بيانات: عقار · بيوت · وصلت · PropertyFinder · العنوان الوطني السعودي<br>
        للأغراض التوجيهية فقط · لا يُعتمد عليه للقرارات المالية النهائية
    </footer>""", unsafe_allow_html=True)
