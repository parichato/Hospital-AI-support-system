# ----------------------------------------------------------
# 🏥 Hospital AI Decision Support System
# Model: CatBoost Multi-Class (Minor / Severe / Fatal)
# Output: เสี่ยงน้อย / เสี่ยงปานกลาง / เสี่ยงมาก + คำแนะนำ
# ----------------------------------------------------------

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

# ----------------------------------------------------------
# ⚙️ Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="Hospital AI Decision Support", page_icon="🏥", layout="wide")
st.title("🏥 Hospital AI Decision Support — Injury Severity (CatBoost)")
st.caption("ใส่ข้อมูลผู้บาดเจ็บ ระบบจะประเมินระดับความรุนแรงและแนะนำขั้นตอนถัดไป")

# ----------------------------------------------------------
# 📦 Load Model + Encoders + Features (แบบไม่เด่น)
# ----------------------------------------------------------
@st.cache_resource
def load_all():
    msgs = []

    try:
        model = joblib.load("predict_catboost_multi.pkl")
        msgs.append("✅ Loaded: predict_catboost_multi.pkl")
    except:
        model = None
        msgs.append("❌ ไม่พบ predict_catboost_multi.pkl")

    try:
        encoders = joblib.load("encoders_multi.pkl")
        msgs.append("✅ Loaded: encoders_multi.pkl")
    except:
        encoders = None
        msgs.append("⚠️ ไม่พบ encoders_multi.pkl")

    try:
        with open("features_multi.json", "r") as f:
            features = json.load(f)
        msgs.append("✅ Loaded: features_multi.json")
    except:
        features = []
        msgs.append("⚠️ ไม่พบ features_multi.json")

    # แสดงผลแบบเรียบใน expander
    with st.expander("📂 รายการไฟล์ที่โหลดแล้ว", expanded=False):
        for m in msgs:
            st.caption(m)

    return model, encoders, features

model, encoders, features = load_all()

# ----------------------------------------------------------
# 🗺️ Manual Mapping (แทน Data Dictionary)
# ----------------------------------------------------------
activity_mapping = {
    "0": "เดินเท้า",
    "1": "โดยสารพาหนะสาธารณะ",
    "2": "โดยสารพาหนะส่วนบุคคล",
    "3": "ขับขี่พาหนะส่วนบุคคล",
    "4": "ทำงาน",
    "5": "เล่นกีฬา",
    "6": "กิจกรรมอื่น ๆ"
}

aplace_mapping = {
    "10": "บ้านพักอาศัย",
    "11": "ถนน/ทางหลวง",
    "12": "สถานที่ทำงาน",
    "13": "โรงเรียน/สถาบันศึกษา",
    "14": "พื้นที่สาธารณะ",
    "15": "อื่น ๆ"
}

prov_mapping = {
    "10": "กรุงเทพมหานคร",
    "20": "เชียงใหม่",
    "30": "ขอนแก่น",
    "40": "ภูเก็ต",
    "50": "นครราชสีมา",
    "60": "สงขลา",
    "99": "อื่น ๆ"
}

# ----------------------------------------------------------
# 🔍 Severity Mapping & Advice
# ----------------------------------------------------------
severity_map = {
    0: "เสี่ยงน้อย",
    1: "เสี่ยงปานกลาง",
    2: "เสี่ยงมาก"
}

advice_map = {
    "เสี่ยงน้อย": "เฝ้าดูอาการ จัดการบาดแผลพื้นฐาน ประเมินซ้ำทุก 15–30 นาที",
    "เสี่ยงปานกลาง": "ส่งตรวจเพิ่มเติม ให้สารน้ำ/ยาแก้ปวด ติดตามสัญญาณชีพใกล้ชิด",
    "เสี่ยงมาก": "แจ้งทีมสหสาขา เปิดทางเดินหายใจ เตรียมห้องฉุกเฉินหรือส่งต่อด่วน"
}

# ----------------------------------------------------------
# 🧩 Input Form
# ----------------------------------------------------------
st.subheader("📋 ข้อมูลผู้บาดเจ็บ")

with st.form("input_form"):
    age = st.number_input("อายุ (ปี)", min_value=0, max_value=120, value=35)
    sex = st.radio("เพศ", ["หญิง", "ชาย"], horizontal=True)
    is_night = st.checkbox("เกิดเหตุเวลากลางคืน", value=False)
    head_injury = st.checkbox("บาดเจ็บที่ศีรษะ", value=False)
    mass_casualty = st.checkbox("เหตุการณ์ผู้บาดเจ็บจำนวนมาก", value=False)

    st.markdown("**ปัจจัยเสี่ยง (Risk Factors)**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: risk1 = st.checkbox("ไม่สวมหมวกนิรภัย / เข็มขัดนิรภัย")
    with c2: risk2 = st.checkbox("ขับรถเร็ว / ประมาท")
    with c3: risk3 = st.checkbox("เมา / ดื่มสุรา")
    with c4: risk4 = st.checkbox("ผู้สูงอายุ / เด็กเล็ก")
    with c5: risk5 = st.checkbox("บาดเจ็บหลายตำแหน่ง")

    st.markdown("**สารเสพติด/ยาในร่างกาย**")
    d1, d2, d3 = st.columns(3)
    with d1: cannabis = st.checkbox("กัญชา")
    with d2: amphetamine = st.checkbox("ยาบ้า / แอมเฟตามีน")
    with d3: drugs = st.checkbox("ยาอื่น ๆ")

    st.markdown("**บริบทของเหตุการณ์**")
    activity = st.selectbox("กิจกรรมขณะเกิดเหตุ", list(activity_mapping.values()))
    aplace = st.selectbox("สถานที่เกิดเหตุ", list(aplace_mapping.values()))
    prov = st.selectbox("จังหวัดที่เกิดเหตุ", list(prov_mapping.values()))

    submit = st.form_submit_button("🔎 ประเมินระดับความเสี่ยง")

# ----------------------------------------------------------
# 🔄 Preprocess Function
# ----------------------------------------------------------
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    reverse_activity = {v: k for k, v in activity_mapping.items()}
    reverse_aplace = {v: k for k, v in aplace_mapping.items()}
    reverse_prov = {v: k for k, v in prov_mapping.items()}

    if df.at[0, "activity"] in reverse_activity:
        df.at[0, "activity"] = reverse_activity[df.at[0, "activity"]]
    if df.at[0, "aplace"] in reverse_aplace:
        df.at[0, "aplace"] = reverse_aplace[df.at[0, "aplace"]]
    if df.at[0, "prov"] in reverse_prov:
        df.at[0, "prov"] = reverse_prov[df.at[0, "prov"]]

    for col in [
        "age", "sex", "is_night", "head_injury", "mass_casualty",
        "risk1", "risk2", "risk3", "risk4", "risk5",
        "cannabis", "amphetamine", "drugs"
    ]:
        df[col] = df[col].astype(float)

    for col in ["activity", "aplace", "prov"]:
        val = str(df.at[0, col])
        if encoders and col in encoders:
            le = encoders[col]
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0
        else:
            df[col] = 0

    df["age_group_60plus"] = (df["age"] >= 60).astype(int)
    df["risk_count"] = df[["risk1","risk2","risk3","risk4","risk5"]].sum(axis=1)
    df["night_flag"] = df["is_night"].astype(int)

    df = df.reindex(columns=features, fill_value=0)
    return df

# ----------------------------------------------------------
# 🧠 Run Prediction (Text Only)
# ----------------------------------------------------------
if submit:
    input_data = {
        "age": age,
        "sex": 1 if sex == "ชาย" else 0,
        "is_night": int(is_night),
        "head_injury": int(head_injury),
        "mass_casualty": int(mass_casualty),
        "risk1": int(risk1),
        "risk2": int(risk2),
        "risk3": int(risk3),
        "risk4": int(risk4),
        "risk5": int(risk5),
        "cannabis": int(cannabis),
        "amphetamine": int(amphetamine),
        "drugs": int(drugs),
        "activity": activity,
        "aplace": aplace,
        "prov": prov
    }

    X_input = preprocess_input(input_data)

    if model is not None:
        probs = model.predict_proba(X_input)[0]
        pred_class = int(np.argmax(probs))
        label = severity_map.get(pred_class, "ไม่ทราบ")

        st.markdown(f"### 🩺 ระดับความเสี่ยงที่คาดการณ์: **{label}**")
        st.info(f"💡 คำแนะนำเบื้องต้น: {advice_map[label]}")
        st.caption(f"🧠 ความมั่นใจของโมเดล: {probs[pred_class]*100:.1f}%")
    else:
        st.error("⚠️ ไม่พบโมเดล ไม่สามารถทำนายได้")
