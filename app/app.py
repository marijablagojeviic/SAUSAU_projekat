import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Dodaj putanju do src foldera
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_single, load_feature_columns

st.set_page_config(
    page_title="BEMS - Predikcija potrošnje energije",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 BEMS - Predikcija potrošnje energije")
st.write("Unesite podatke o zgradi za predikciju potrošnje")

# Učitaj feature kolone
feature_columns = load_feature_columns()

st.info(f"💡 Potrebni feature-i: {', '.join(feature_columns)}")

# Kreiraj dva stupca za unos
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Podaci o zgradi")
    site_id = st.number_input("Site ID", value=0.0, step=1.0)
    primary_use = st.number_input("Primary Use (encoded)", value=0.0, step=1.0)
    square_feet = st.number_input("Kvadratura (sq ft)", value=10000.0)
    floor_count = st.number_input("Broj spratova", value=5.0)

with col2:
    st.subheader("🌤️ Vremenski podaci")
    air_temperature = st.number_input("Temperatura (°C)", value=20.0)
    cloud_coverage = st.number_input("Oblačnost (0-1)", value=0.5)
    dew_temperature = st.number_input("Tačka rose (°C)", value=10.0)
    wind_speed = st.number_input("Brzina vjetra (m/s)", value=3.0)

col3, col4 = st.columns(2)

with col3:
    st.subheader("🕐 Vremenske komponente")
    hour = st.slider("Sat", 0, 23, 12)
    day_of_week = st.slider("Dan u sedmici (0=Pon, 6=Ned)", 0, 6, 0)

with col4:
    st.subheader("⚙️ Ostalo")
    month = st.slider("Mjesec", 1, 12, 1)
    is_business_hour = st.selectbox(
        "Radno vrijeme", 
        [0, 1], 
        format_func=lambda x: "Da" if x == 1 else "Ne"
    )
    tariff_zone = st.selectbox(
        "Tarifna zona", 
        [0, 1], 
        format_func=lambda x: "Viša" if x == 1 else "Niža"
    )

# Dugme za predikciju
if st.button("🔮 Predvidi potrošnju", type="primary"):
    # Skupi sve feature-e
    features = {
        'site_id': site_id,
        'primary_use': primary_use,
        'square_feet': square_feet,
        'floor_count': floor_count,
        'air_temperature': air_temperature,
        'cloud_coverage': cloud_coverage,
        'dew_temperature': dew_temperature,
        'wind_speed': wind_speed,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_business_hour': is_business_hour,
        'tariff_zone': tariff_zone
    }
    
    try:
        # Pozovi predikciju (koristi Random Forest model)
        prediction = predict_single(features, 'rf_model')
        
        # Prikaži rezultat
        st.success(f"✅ Predviđena potrošnja: {prediction:.2f} kWh")
        
        # BEMS logika
        st.subheader("📊 BEMS Analiza")
        
        if air_temperature > 24 and prediction > 150:
            st.warning("🔥 PRE-COOL - Aktivno hlađenje")
            st.write("Preporuka: Uključiti sistem za hlađenje prije dolaska toplotnog talasa.")
        elif air_temperature < 10:
            st.warning("❄️ PRE-HEAT - Aktivno grijanje")
            st.write("Preporuka: Uključiti sistem za grijanje prije dolaska hladnog talasa.")
        else:
            st.info("💡 STANDBY - Normalan režim rada")
            st.write("Sistem radi u normalnom režimu, nisu potrebne dodatne akcije.")
        
        # Dodatne informacije
        with st.expander("📋 Detaljne informacije"):
            st.write("**Uneseni parametri:**")
            for key, value in features.items():
                st.write(f"- {key}: {value}")
            
            st.write("**Korišteni model:** Random Forest")
            
    except Exception as e:
        st.error(f"❌ Greška: {e}")
        st.info("Provjeri da li su svi modeli istrenirani!")