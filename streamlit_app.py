import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os
import re

st.set_page_config(
    page_title="NutriSense - Pantau Pertumbuhan Anak", 
    page_icon="🍎",
    layout="wide"
)

def load_css():
    css_file = "style.css"
    if os.path.exists(css_file):
        with open(css_file, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_html_template(template_name):
    html_file = "index.html"
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = f'<template id="{template_name}">(.*?)</template>'
    match = re.search(pattern, content, re.DOTALL)
    
    return match.group(1).strip()

def render_header():
    col_logo, col_title = st.columns([2, 8])
    
    with col_logo:
        if os.path.exists("asset/logo.png"):
            st.image("asset/logo.png", width=100)
    
    with col_title:
        header_html = load_html_template("header-template")
        st.markdown(header_html, unsafe_allow_html=True)

def interpolate(x1, y1, x2, y2, x):
    return y1 + (y2 - y1) * ((x - x1) / (x2 - x1))

def hitung_z_tb_u(age_month, height, sex, who_df):
    data = who_df[who_df["sex"] == sex].sort_values("age_month")
    
    if data.empty:
        return None
    
    if age_month in data["age_month"].values:
        row = data[data["age_month"] == age_month].iloc[0]
        return (height - row["median"]) / row["sd"]
    
    lower = data[data["age_month"] < age_month].tail(1)
    upper = data[data["age_month"] > age_month].head(1)
    
    if lower.empty or upper.empty:
        return None
    
    a1, m1, s1 = lower.iloc[0][["age_month", "median", "sd"]]
    a2, m2, s2 = upper.iloc[0][["age_month", "median", "sd"]]
    
    median = interpolate(a1, m1, a2, m2, age_month)
    sd = interpolate(a1, s1, a2, s2, age_month)
    
    return (height - median) / sd

def status_who(z_tb):
    if z_tb < -3:
        return "Tinggi anak jauh di bawah standar WHO. Sangat disarankan untuk segera konsultasi ke tenaga kesehatan."
    elif z_tb < -2:
        return "Tinggi anak berada di bawah standar WHO. Disarankan konsultasi dengan tenaga kesehatan."
    else:
        return "Tinggi anak sesuai dengan standar pertumbuhan WHO."

def save_history(child_name, age_month, sex, height, weight, who_status):
    data = {
        "child_name": child_name.strip().lower(),
        "age_month": age_month,
        "sex": sex,
        "height": height,
        "weight": weight,
        "who_status": who_status,
        "timestamp": datetime.now()
    }
    
    df_new = pd.DataFrame([data])
    
    if os.path.exists("child_history.csv"):
        df_old = pd.read_csv("child_history.csv")
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    
    df.to_csv("child_history.csv", index=False)

def load_child_history(child_name):
    if not os.path.exists("child_history.csv"):
        return pd.DataFrame()
    
    df = pd.read_csv("child_history.csv")
    return df[df["child_name"] == child_name.strip().lower()]

def plot_growth_with_history(child_name, current_age, current_height, sex, who_df, history_df):
    who_data = who_df[who_df["sex"] == sex].sort_values("age_month")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(
        who_data["age_month"],
        who_data["median"],
        color="#B8B8B8",
        linewidth=3,
        linestyle='--',
        label="Standar WHO",
        alpha=0.7
    )

    ax.plot(
        who_data["age_month"],
        who_data["median"] - 2 * who_data["sd"],
        color="#FF9800",
        linewidth=2.5,
        linestyle='--',
        label="Berisiko Stunting",
        alpha=0.7
    )
    
    ax.plot(
        who_data["age_month"],
        who_data["median"] - 3 * who_data["sd"],
        color="#F44336",
        linewidth=2.5,
        linestyle='--',
        label="Stunting",
        alpha=0.7
    )

    ax.plot(
        who_data["age_month"],
        who_data["median"] + 2 * who_data["sd"],
        color="#AB47BC",
        linewidth=2.5,
        linestyle='--',
        label="Berisiko Obesitas",
        alpha=0.7
    )
    
    ax.plot(
        who_data["age_month"],
        who_data["median"] + 3 * who_data["sd"],
        color="#7B1FA2",
        linewidth=2.5,
        linestyle='--',
        label="Obesitas",
        alpha=0.7
    )
    
    if not history_df.empty:
        history_sorted = history_df.sort_values("age_month")
        ax.plot(
            history_sorted["age_month"],
            history_sorted["height"],
            color="#87CEEB",
            linewidth=3,
            marker='o',
            markersize=10,
            label=f"Riwayat {child_name.title()}",
            markeredgecolor='white',
            markeredgewidth=2
        )
    
    ax.scatter(
        current_age,
        current_height,
        color="#FF6B6B",
        s=300,
        zorder=5,
        label="Data Terbaru",
        edgecolors='white',
        linewidths=3
    )
    
    ax.set_xlabel("Usia (bulan)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Tinggi Badan (cm)", fontsize=14, fontweight='bold')
    ax.set_title(f"Grafik Pertumbuhan {child_name.title()}", fontsize=18, fontweight='bold', pad=20)
    
    ax.legend(loc="upper left", fontsize=12, frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')
    
    return fig

def render_result_cards(pred_label, pred_proba, record_count):
    col1, col2, col3 = st.columns(3)
    
    card_template = load_html_template("result-card-template")
    
    with col1:
        status_color = "#4CAF50" if pred_label == "Normal" else "#FF9800" if pred_label == "Berisiko" else "#F44336"
        html = card_template.replace("{title}", "Status Gizi").replace("{color}", status_color).replace("{value}", pred_label)
        st.markdown(html, unsafe_allow_html=True)
    
    #pakk, tingkat keyakinan ini maksudnya kemungkinan dia akan kena stunting yaa~
    with col2: 
        html = card_template.replace("{title}", "Tingkat Keyakinan").replace("{color}", "#2196F3").replace("{value}", f"{pred_proba*100:.1f}%")
        st.markdown(html, unsafe_allow_html=True)
    
    with col3:
        html = card_template.replace("{title}", "Total Pemeriksaan").replace("{color}", "#9C27B0").replace("{value}", f"{record_count}x")
        st.markdown(html, unsafe_allow_html=True)

def render_who_result(z_tb, who_status):
    if z_tb < -2:
        template = load_html_template("who-warning-template")
    else:
        template = load_html_template("who-success-template")
    
    html = template.replace("{status}", who_status)
    st.markdown(html, unsafe_allow_html=True)

def render_spacer():
    spacer_html = load_html_template("spacer-template")
    st.markdown(spacer_html, unsafe_allow_html=True)

def render_caption(text):
    caption_template = load_html_template("caption-template")
    html = caption_template.replace("{text}", text)
    st.markdown(html, unsafe_allow_html=True)

def main():
    load_css()
    
    if "reset_form" not in st.session_state:
        st.session_state.reset_form = False
    
    who_df = pd.read_csv("malnutrition_data_who.csv")
    who_df = who_df.sort_values("age_month")
    model = joblib.load("nutrisense_model.pkl")
    
    render_header()
    render_spacer()
    
    with st.form(key="input_form"):
        child_name = st.text_input("Nama Anak", placeholder="Masukkan nama anak")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_month = st.number_input("Usia (bulan)", min_value=0, max_value=60, value=0)
        
        with col2:
            height = st.number_input("Tinggi badan (cm)", min_value=0.0, max_value=150.0, value=0.0, step=0.1)
        
        with col3:
            weight = st.number_input("Berat badan (kg)", min_value=0.0, max_value=30.0, value=0.0, step=0.1)
        
        sex = st.radio(
            "Jenis Kelamin",
            options=["Male", "Female"],
            format_func=lambda x: "Laki-laki" if x == "Male" else "Perempuan",
            horizontal=True
        )
        
        submit_button = st.form_submit_button("Cek Pertumbuhan")
    
    if not child_name.strip() and not submit_button:
        st.info("Masukkan data anak untuk memulai pemeriksaan")
        st.stop()
    
    if submit_button:
        if not child_name.strip():
            st.warning("Mohon isi nama anak terlebih dahulu")
            st.stop()
        
        history_df = load_child_history(child_name)
        
        age = int(round(age_month / 12))
        X_new = pd.DataFrame([[age, weight, height]], columns=["age", "weight", "height"])
        pred_encoded = model.predict(X_new)[0]
        pred_proba = model.predict_proba(X_new).max()
        
        label_map_rev = {0: "Normal", 1: "Berisiko", 2: "Stunting"}
        pred_label = label_map_rev[pred_encoded]
        
        render_spacer()
        st.subheader("Hasil Pemeriksaan")
        
        record_count = len(history_df) + 1
        render_result_cards(pred_label, pred_proba, record_count)
        
        z_tb = hitung_z_tb_u(age_month, height, sex, who_df)
        
        if z_tb is not None:
            who_status = status_who(z_tb)
            
            save_history(child_name, age_month, sex, height, weight, who_status)
            
            updated_history = load_child_history(child_name)
            
            render_spacer()
            with st.container():
                st.subheader("Grafik Pertumbuhan")
                fig = plot_growth_with_history(child_name, age_month, height, sex, who_df, updated_history)
                st.pyplot(fig, use_container_width=True)
            
            render_spacer()
            render_who_result(z_tb, who_status)
        else:
            st.error("Data WHO tidak tersedia untuk usia ini")
        
        render_caption("Catatan: Grafik pertumbuhan mengacu pada standar WHO. Prediksi status gizi menggunakan model Machine Learning berdasarkan data historis.")

if __name__ == "__main__":
    main()