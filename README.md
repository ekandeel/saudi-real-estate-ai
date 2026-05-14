# 🏠 Saudi Real Estate AI — مستشار العقار السعودي الذكي

> **Proof of Concept:** Transforming a traditional real estate database into an AI-powered pricing model — from raw data to a live web application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 🎯 Project Goal

This project demonstrates how **traditional real estate data can be transformed into an intelligent AI product** — covering the full pipeline:

```
Raw Data → Cleaning → Feature Engineering → ML Model → REST API → Web App
```

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| Total records | 127,053 listings |
| Sources | Aqar · Bayut · Wasalt · PropertyFinder |
| Cities | 791 Saudi cities |
| GPS coverage | 99% |
| Date range | Jul 2024 – May 2025 |

---

## 🤖 Models

### Sale Price Model (XGBoost)
- **R² = 0.860** on 77,293 records
- MAE = 2,229 SAR/m²
- Features: area, city, neighborhood, rooms, baths, elevator, parking, AC, furnished

### Rent Price Model (XGBoost)  
- **R² = 0.604** on 32,835 records
- Needs more data for improvement (honest assessment)

---

## 🔑 Key Insights from Data

- 📈 Market grew **×8.7** in 10 months (Jul 2024 → Apr 2025)
- 🏙️ Riyadh = **45%** of all listings, median price 1.3M SAR
- 💰 Golden range **500K–1M SAR** = 36% of market
- 📍 99% of listings have GPS coordinates (rare for Arabic real estate data)

---

## 🚀 Live Demo

```bash
git clone https://github.com/ekandeel/saudi-real-estate-ai
cd saudi-real-estate-ai
pip install -r requirements.txt
streamlit run app_full.py
```

Open: `http://localhost:8501`

---

## 🛠️ Tech Stack

```
Python 3.12
├── XGBoost 3.2      — Price prediction model
├── Scikit-learn     — Preprocessing & evaluation  
├── Pandas / NumPy   — Data pipeline
└── Streamlit        — Web application
```

---

## 📁 File Structure

```
saudi-real-estate-ai/
├── app_full.py                    # Streamlit web app (Arabic RTL)
├── xgb_sale_full.pkl              # Trained sale model
├── xgb_rent_full.pkl              # Trained rent model
├── city_neighborhoods_full.json   # 791 cities × neighborhoods mapping
├── neighborhood_stats_full.csv    # Market stats per neighborhood
├── requirements.txt
└── README.md
```

---

## 💼 CV Summary

> *"Built an end-to-end AI pipeline on 127K real Saudi real estate listings — data collection from 4 platforms, XGBoost pricing model (R²=0.86), and Arabic-language web application deployed on Streamlit Cloud."*

---

## 🤝 Looking for a Saudi Partner

This is a Proof of Concept. I'm looking for a Saudi real estate professional to:
- Expand the dataset with live scraping
- Validate model accuracy against real market knowledge  
- Co-develop a full Flutter mobile app (Android + iOS)

**Contact:** Ekandeel@gmail.com

---

*Built with real Saudi real estate data · May 2025*
