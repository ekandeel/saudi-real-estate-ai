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
    border-radius:16px;padding:24px 28px;text-align:center;margin:12px 0;color:white;
}
.result-price{font-size:36px;font-weight:700;margin:0;}
.result-label{font-size:13px;opacity:.85;margin:0 0 6px;}
.result-range{font-size:13px;opacity:.75;margin:5px 0 0;}
.mcard{border-radius:10px;padding:10px 14px;margin:5px 0;border-right:4px solid #1a4e8f;background:#f0f4fd;}
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
.report-box{
    background:#f8f9fb;border:1px solid #e2ddd5;border-radius:12px;
    padding:20px 24px;margin-top:16px;font-size:13px;line-height:2;
    direction:rtl;text-align:right;
}
.report-box h4{font-size:15px;font-weight:700;margin-bottom:8px;color:#1a4e8f;}
.report-section{margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid #e2ddd5;}
.report-section:last-child{border-bottom:none;margin-bottom:0;}
footer{text-align:center;color:#999;font-size:11px;margin-top:24px;}
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
    cent  = load_centroids()
    posts = load_posts_geo()
    return cKDTree(cent[['lat','lng']].values), cKDTree(posts[['lat','lng']].values)

models    = load_models()
hier      = load_hierarchy()
stats     = load_stats()
posts_geo = load_posts_geo()
cent_df   = load_centroids()
cent_tree, posts_tree = build_trees()

# ── Session state ─────────────────────────────────────────────────────────────
for k,v in {
    'sel_region':'منطقة الرياض','sel_city':'الرياض',
    'sel_district':'الملقا','sel_lat':24.812,'sel_lng':46.698,
    'map_center':[24.7136,46.6753],'map_zoom':11,
}.items():
    if k not in st.session_state: st.session_state[k]=v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏠 مستشار العقار السعودي الذكي")
c1,c2,c3 = st.columns(3)
with c1: st.markdown('<span class="badge bb">127,000+ إعلان</span>',unsafe_allow_html=True)
with c2: st.markdown('<span class="badge bg">R² بيع = 0.875</span>',unsafe_allow_html=True)
with c3: st.markdown('<span class="badge ba">العنوان الوطني السعودي</span>',unsafe_allow_html=True)
st.divider()

# ═══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════
left_col, map_col = st.columns([1, 1.8])

# ── LEFT: Controls ────────────────────────────────────────────────
with left_col:
    ltype_label = st.radio("نوع العملية", ["بيع","إيجار"], horizontal=True)
    ltype = 'sale' if ltype_label=='بيع' else 'rent'
    st.markdown("---")

    # Location tabs
    tab_list, tab_map = st.tabs(["📋 اختر من القائمة", "📡 انقر على الخريطة"])

    with tab_list:
        regions = sorted(hier.keys())
        reg_idx = regions.index(st.session_state['sel_region']) if st.session_state['sel_region'] in regions else 0
        sel_region = st.selectbox("🗺️ المنطقة", regions, index=reg_idx, key='dd_region')

        cities = sorted(hier.get(sel_region,{}).keys())
        cur_city = st.session_state['sel_city'] if st.session_state['sel_city'] in cities else (cities[0] if cities else '')
        city_idx = cities.index(cur_city) if cur_city in cities else 0
        sel_city = st.selectbox("🏙️ المدينة", cities, index=city_idx, key='dd_city')

        districts = hier.get(sel_region,{}).get(sel_city,[])
        cur_dist  = st.session_state['sel_district'] if st.session_state['sel_district'] in districts else (districts[0] if districts else None)
        dist_idx  = districts.index(cur_dist) if cur_dist in districts else 0
        sel_district = st.selectbox("🏘️ الحي", districts, index=dist_idx, key='dd_district') if districts else None

        # Sync to map
        if sel_district:
            row = cent_df[(cent_df['city_ar']==sel_city)&(cent_df['district_ar']==sel_district)]
            if not row.empty:
                st.session_state.update({
                    'sel_lat': row.iloc[0]['lat'], 'sel_lng': row.iloc[0]['lng'],
                    'sel_region': sel_region, 'sel_city': sel_city,
                    'sel_district': sel_district,
                    'map_center': [row.iloc[0]['lat'], row.iloc[0]['lng']],
                    'map_zoom': 14,
                })
        elif sel_city:
            city_rows = cent_df[cent_df['city_ar']==sel_city]
            if not city_rows.empty:
                st.session_state['map_center'] = [city_rows['lat'].mean(), city_rows['lng'].mean()]
                st.session_state['map_zoom']   = 12

    with tab_map:
        st.caption("انقر على الخريطة — سيتم تحديد الحي تلقائياً")
        if st.session_state.get('last_click_info'):
            st.markdown(f"""
            <div class="mcard green">
                ✅ <strong>{st.session_state['sel_city']}</strong>
                ← حي <strong>{st.session_state['sel_district']}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("لم يتم النقر على الخريطة بعد")

    # Selected location summary
    sd = st.session_state['sel_district']
    sc = st.session_state['sel_city']
    sr = st.session_state['sel_region']
    if sd:
        st.markdown(f"""
        <div class="mcard green" style="margin-top:8px">
            📍 <strong>{sr}</strong> ← {sc} ← حي <strong>{sd}</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🏗️ تفاصيل العقار**")

    area = st.number_input("📐 المساحة (م²)", min_value=20, max_value=50000, value=150, step=10)
    area_tol = st.number_input("نطاق المساحة ± م²", min_value=0, max_value=100, value=10, step=5)
    st.caption(f"البحث: {area-area_tol} م² ← {area+area_tol} م²")
    st.caption(f"البحث: {area-area_tol} م² ← {area+area_tol} م²")

    d1,d2 = st.columns(2)
    with d1: rooms = st.number_input("🛏️ الغرف",    1, 12, 3, 1)
    with d2: baths = st.number_input("🚿 الحمامات", 1, 10, 2, 1)

    radius_km = st.number_input("🔍 نطاق البحث كم", 1, 15, 3, 1)

    st.markdown("**✨ المرافق:**")
    a1,a2 = st.columns(2)
    with a1:
        has_parking  = st.checkbox("🚗 موقف سيارة", value=True)
        has_ac       = st.checkbox("❄️ مكيف مركزي", value=False)
    with a2:
        is_furnished = st.checkbox("🛋️ مفروش", value=False)

    st.markdown("---")
    predict_btn = st.button("🔍 تقييم العقار", use_container_width=True, type="primary")

# ── RIGHT: Map ────────────────────────────────────────────────────
with map_col:
    st.markdown("**🗺️ خريطة العقارات**")

    sel_lat = st.session_state['sel_lat']
    sel_lng = st.session_state['sel_lng']
    sel_city     = st.session_state['sel_city']
    sel_district = st.session_state['sel_district']
    sel_region   = st.session_state['sel_region']

    m = folium.Map(
        location=st.session_state['map_center'],
        zoom_start=st.session_state['map_zoom'],
        tiles='CartoDB positron'
    )

    # District centroids for current city
    city_cents = cent_df[cent_df['city_ar']==sel_city]
    for _, cr in city_cents.iterrows():
        is_sel = (sel_district and cr['district_ar']==sel_district)
        folium.CircleMarker(
            location=[cr['lat'], cr['lng']],
            radius=7 if is_sel else 4,
            color='#c0392b' if is_sel else '#2d5fa6',
            fill=True, fill_opacity=0.85 if is_sel else 0.35,
            weight=2 if is_sel else 1,
            tooltip=f"حي {cr['district_ar']} | {int(cr['post_count'])} إعلان"
        ).add_to(m)

    # Nearby listings with area filter
    radius_deg = radius_km / 111.0
    nb_idx  = posts_tree.query_ball_point([sel_lat, sel_lng], r=radius_deg)
    nearby  = posts_geo.iloc[nb_idx].copy()
    nearby_ltype = nearby[
        (nearby['listing_type_key']==ltype) &
        (nearby['area_clean'] >= area - area_tol) &
        (nearby['area_clean'] <= area + area_tol)
    ]
    nearby_all_type = nearby[nearby['listing_type_key']==ltype]

    # Show all type listings (grey) and area-matched (green)
    for _, pr in nearby_all_type.head(200).iterrows():
        matched = (abs(pr['area_clean'] - area) <= area_tol) if pd.notna(pr['area_clean']) else False
        folium.CircleMarker(
            location=[pr['lat'], pr['lng']],
            radius=5 if matched else 3,
            color='#27ae60' if matched else '#95a5a6',
            fill=True, fill_opacity=0.7 if matched else 0.4,
            weight=1,
            tooltip=f"{'✅ ' if matched else ''}{pr['price_per_sqm']:,.0f} ر/م² | {pr['area_clean']:.0f}م²"
        ).add_to(m)

    # Search circle
    folium.Circle(
        [sel_lat, sel_lng], radius=radius_km*1000,
        color='#2d5fa6', fill=True, fill_opacity=0.04,
        weight=1.5, dash_array='6'
    ).add_to(m)

    # Selected marker
    if sel_district:
        folium.Marker(
            [sel_lat, sel_lng],
            tooltip=f"📍 {sel_district}",
            icon=folium.Icon(color='red', icon='home', prefix='fa')
        ).add_to(m)

    # Legend
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:20px;right:20px;z-index:1000;
        background:white;padding:10px 14px;border-radius:8px;
        border:1px solid #ddd;font-size:12px;direction:rtl;min-width:170px;">
        <div style="font-weight:600;margin-bottom:5px">دليل الألوان</div>
        <div>🔴 الحي المختار</div>
        <div>🔵 أحياء المدينة</div>
        <div>🟢 إعلانات مطابقة للمساحة ({len(nearby_ltype)})</div>
        <div>⚪ إعلانات أخرى قريبة ({len(nearby_all_type)-len(nearby_ltype)})</div>
        <div style="margin-top:4px;color:#666">نطاق: {radius_km} كم | مساحة: {area-area_tol}–{area+area_tol} م²</div>
    </div>
    """))

    map_data = st_folium(m, height=530, use_container_width=True,
                         returned_objects=["last_clicked"])

    # Handle map click
    if map_data and map_data.get("last_clicked"):
        clk = map_data["last_clicked"]
        d_arr, i_arr = cent_tree.query([[clk["lat"], clk["lng"]]], k=1)
        dist_km_c = float(d_arr.ravel()[0]) * 111
        nr = cent_df.iloc[int(i_arr.ravel()[0])]
        st.session_state.update({
            'sel_lat': clk["lat"], 'sel_lng': clk["lng"],
            'sel_region': nr['region_ar'], 'sel_city': nr['city_ar'],
            'sel_district': nr['district_ar'],
            'map_center': [clk["lat"], clk["lng"]],
            'map_zoom': 14, 'last_click_info': True,
        })
        st.rerun()

# ═══════════════════════════════════════════════════════════════════
# PREDICTION RESULTS
# ═══════════════════════════════════════════════════════════════════
if predict_btn:
    sd = st.session_state['sel_district']
    sc = st.session_state['sel_city']
    sr = st.session_state['sel_region']
    sl, sg = st.session_state['sel_lat'], st.session_state['sel_lng']

    if not sd:
        st.error("الرجاء اختيار حي أولاً.")
        st.stop()
    if ltype not in models:
        st.error("النموذج غير متوفر.")
        st.stop()

    pkg     = models[ltype]
    model   = pkg['model']
    le_city = pkg['le_city']
    le_nbhd = pkg['le_nbhd']

    city_enc = le_city.transform([sc])[0] if sc in le_city.classes_ else 0
    nbhd_enc = le_nbhd.transform([sd])[0] if sd in le_nbhd.classes_ else 0

    X_in = pd.DataFrame([{
        'city_enc':city_enc, 'nbhd_enc':nbhd_enc,
        'area_clean':area, 'rooms_clean':rooms, 'baths_clean':baths,
        'has_elevator':0,
        'has_parking':int(has_parking), 'has_ac':int(has_ac),
        'is_furnished':int(is_furnished),
    }])

    ml_ppsqm = np.expm1(model.predict(X_in)[0])

    # Nearby with area filter
    r_deg = radius_km / 111.0
    nb_i  = posts_tree.query_ball_point([sl, sg], r=r_deg)
    nb_df = posts_geo.iloc[nb_i]
    nb_t  = nb_df[
        (nb_df['listing_type_key']==ltype) &
        (nb_df['area_clean'] >= area - area_tol) &
        (nb_df['area_clean'] <= area + area_tol)
    ]
    nb_all = nb_df[nb_df['listing_type_key']==ltype]

    if len(nb_t) >= 5:
        nb_med = nb_t['price_per_sqm'].median()
        final_ppsqm = ml_ppsqm * 0.60 + nb_med * 0.40
        nb_note = f"مدمج مع {len(nb_t)} عقار مطابق للمساحة ضمن {radius_km} كم"
    elif len(nb_all) >= 5:
        nb_med = nb_all['price_per_sqm'].median()
        final_ppsqm = ml_ppsqm * 0.70 + nb_med * 0.30
        nb_note = f"مدمج مع {len(nb_all)} عقار قريب (مساحات متفاوتة)"
    else:
        nb_med = None
        final_ppsqm = ml_ppsqm
        nb_note = "نموذج AI فقط — لا توجد إعلانات قريبة كافية"

    pred_total = final_ppsqm * area
    low_total  = final_ppsqm * 0.85 * area
    high_total = final_ppsqm * 1.15 * area

    dist_row  = stats[(stats['city']==sc)&(stats['district']==sd)&(stats['ltype']==ltype)]
    city_rows = stats[(stats['city']==sc)&(stats['ltype']==ltype)]
    city_med  = city_rows['median_ppsqm'].median() if not city_rows.empty else None
    dist_med  = dist_row.iloc[0]['median_ppsqm'] if not dist_row.empty else None
    dist_cnt  = int(dist_row.iloc[0]['count']) if not dist_row.empty else 0

    # ── Metrics row ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 نتائج التقييم")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("سعر/م² المتوقع", f"{final_ppsqm:,.0f} ريال")
    m2.metric("الإجمالي", f"{pred_total:,.0f} ريال")
    m3.metric("وسيط الحي", f"{dist_med:,.0f}" if dist_med else "—")
    m4.metric("إعلانات مطابقة", str(len(nb_t)))

    # ── Cards row ──────────────────────────────────────────────────
    ra, rb, rc = st.columns([1.2, 1, 1])

    with ra:
        st.markdown(f"""
        <div class="result-box">
            <p class="result-label">
                {'🏠 سعر المتر — بيع' if ltype=='sale' else '🔑 إيجار سنوي/م²'}
                &nbsp;·&nbsp; {sr}<br>{sc} ← حي {sd}
            </p>
            <p class="result-price">{final_ppsqm:,.0f} ريال/م²</p>
            <p class="result-range">
                إجمالي {area:,} م²:
                <strong>{pred_total:,.0f} ريال</strong>
            </p>
            <p class="result-range">
                نطاق: {low_total:,.0f} — {high_total:,.0f}
            </p>
            <p class="result-range" style="font-size:11px;opacity:.6;margin-top:4px">
                🔬 {nb_note}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with rb:
        st.markdown("**مقارنة بالسوق**")
        if dist_med:
            diff = (final_ppsqm - dist_med) / dist_med * 100
            if   diff >  15: cls,icon,msg = 'red',  '⚠️', f"أعلى من وسيط الحي بـ {diff:.0f}%"
            elif diff < -15: cls,icon,msg = 'green','✅', f"أقل من وسيط الحي بـ {abs(diff):.0f}%"
            else:             cls,icon,msg = 'mcard','📊', "ضمن النطاق الطبيعي"
            st.markdown(f"""
            <div class="mcard {cls}">
                <strong>{icon} {msg}</strong><br>
                <small>وسيط الحي: {dist_med:,.0f} ر/م² · {dist_cnt} إعلان</small>
            </div>""", unsafe_allow_html=True)
        if city_med:
            st.markdown(f"""
            <div class="mcard amber">
                📍 وسيط {sc}: <strong>{city_med:,.0f}</strong> ر/م²
            </div>""", unsafe_allow_html=True)
        if nb_med:
            st.markdown(f"""
            <div class="mcard green">
                🟢 وسيط القريب المطابق: <strong>{nb_med:,.0f}</strong> ر/م²
                <small>({len(nb_t)} إعلان · {area-area_tol}–{area+area_tol} م²)</small>
            </div>""", unsafe_allow_html=True)

        if not city_rows.empty:
            top = city_rows.nlargest(5,'count')[['district','median_ppsqm','count']]
            rows_html = "".join(
                f"<tr><td>{'★ ' if r['district']==sd else ''}{r['district']}</td>"
                f"<td>{r['median_ppsqm']:,.0f}</td><td>{int(r['count']):,}</td></tr>"
                for _,r in top.iterrows()
            )
            st.markdown("**أكثر الأحياء نشاطاً:**")
            st.markdown(f"""<table class="ctable">
              <thead><tr><th>الحي</th><th>وسيط ر/م²</th><th>إعلانات</th></tr></thead>
              <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)

    with rc:
        st.markdown("**تأثير المواصفات**")
        for feat,label in [('has_parking','🚗 موقف'),
                            ('has_ac','❄️ مكيف'),('is_furnished','🛋️ مفروش')]:
            ron  = X_in.copy(); ron[feat]  = 1
            roff = X_in.copy(); roff[feat] = 0
            d    = np.expm1(model.predict(ron)[0]) - np.expm1(model.predict(roff)[0])
            sign = "+" if d >= 0 else ""
            color= "#155724" if d >= 0 else "#721c24"
            st.markdown(
                f'<div class="irow"><span>{label}</span>'
                f'<span style="color:{color};font-weight:500;">{sign}{d:,.0f} ر/م²</span></div>',
                unsafe_allow_html=True
            )

    # ═══════════════════════════════════════════════════════════════
    # TEXT REPORT
    # ═══════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### 📄 التقرير التفصيلي")

    amenities_list = []
    if has_parking:  amenities_list.append("موقف سيارة")
    if has_ac:       amenities_list.append("مكيف مركزي")
    if is_furnished: amenities_list.append("مفروش")
    amenities_str = " · ".join(amenities_list) if amenities_list else "لا يوجد"

    verdict_text = ""
    if dist_med:
        diff2 = (final_ppsqm - dist_med) / dist_med * 100
        if   diff2 >  15: verdict_text = f"السعر المقدّر أعلى من وسيط الحي بنسبة {diff2:.0f}%، يُنصح بمراجعة التفاصيل."
        elif diff2 < -15: verdict_text = f"السعر المقدّر أقل من وسيط الحي بنسبة {abs(diff2):.0f}%، قد يمثل فرصة جيدة."
        else:             verdict_text = f"السعر المقدّر ضمن النطاق الطبيعي لحي {sd} (فرق {diff2:+.0f}%)."

    nearby_summary = f"{len(nb_t)} إعلان مطابق للمساحة ({area-area_tol}–{area+area_tol} م²)" if len(nb_t)>0 else "لا توجد إعلانات مطابقة للمساحة"

    st.markdown(f"""
    <div class="report-box">

      <div class="report-section">
        <h4>📍 بيانات الموقع</h4>
        المنطقة الإدارية: <strong>{sr}</strong><br>
        المدينة: <strong>{sc}</strong><br>
        الحي: <strong>{sd}</strong><br>
        الإحداثيات: {sl:.5f}° ش · {sg:.5f}° ق
      </div>

      <div class="report-section">
        <h4>🏗️ مواصفات العقار</h4>
        نوع العملية: <strong>{'بيع' if ltype=='sale' else 'إيجار'}</strong><br>
        المساحة: <strong>{area} م²</strong>
        (نطاق البحث: {area-area_tol} – {area+area_tol} م²)<br>
        عدد الغرف: <strong>{rooms}</strong> &nbsp;|&nbsp;
        عدد الحمامات: <strong>{baths}</strong><br>
        المرافق: <strong>{amenities_str}</strong>
      </div>

      <div class="report-section">
        <h4>💰 نتيجة التقييم</h4>
        سعر المتر المربع المقدّر: <strong>{final_ppsqm:,.0f} ريال/م²</strong><br>
        السعر الإجمالي المقدّر: <strong>{pred_total:,.0f} ريال</strong><br>
        النطاق المتوقع: {low_total:,.0f} — {high_total:,.0f} ريال<br>
        طريقة الحساب: {nb_note}
      </div>

      <div class="report-section">
        <h4>📊 مقارنة بالسوق</h4>
        وسيط حي {sd}: <strong>{f"{dist_med:,.0f} ريال/م²" if dist_med else "غير متوفر"}</strong>
        {f"({dist_cnt} إعلان)" if dist_cnt else ""}<br>
        وسيط مدينة {sc}: <strong>{f"{city_med:,.0f} ريال/م²" if city_med else "غير متوفر"}</strong><br>
        {f"وسيط الإعلانات القريبة المطابقة: <strong>{nb_med:,.0f} ريال/م²</strong> ({len(nb_t)} إعلان)<br>" if nb_med else ""}
        {f"<strong>الحكم: {verdict_text}</strong>" if verdict_text else ""}
      </div>

      <div class="report-section">
        <h4>🔍 البيانات القريبة</h4>
        نطاق البحث: <strong>{radius_km} كم</strong><br>
        إجمالي الإعلانات القريبة ({ltype_label}): <strong>{len(nb_all)}</strong><br>
        {nearby_summary}<br>
        {f"أدنى سعر قريب: {nb_all['price_per_sqm'].min():,.0f} ريال/م²" if len(nb_all)>0 else ""}<br>
        {f"أعلى سعر قريب: {nb_all['price_per_sqm'].max():,.0f} ريال/م²" if len(nb_all)>0 else ""}
      </div>

      <div class="report-section">
        <h4>ℹ️ ملاحظات</h4>
        — هذا التقرير للأغراض التوجيهية فقط ولا يُعتمد عليه للقرارات المالية النهائية.<br>
        — النموذج مبني على 127,053 إعلان من منصات: عقار، بيوت، وصلت، PropertyFinder.<br>
        — بيانات الأحياء مصدرها العنوان الوطني السعودي (maps.address.gov.sa).<br>
        — دقة نموذج البيع: R² = 0.875 | دقة نموذج الإيجار: R² = 0.674<br>
        — تاريخ البيانات: يوليو 2024 – أبريل 2025.
      </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <footer>
        مستشار العقار السعودي الذكي · بيانات: عقار · بيوت · وصلت · PropertyFinder · العنوان الوطني
    </footer>""", unsafe_allow_html=True)
