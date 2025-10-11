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
st.set_page_config(page_title="AI Injury Severity", page_icon="🏥", layout="wide")
st.title("🏥 Hospital AI Decision Support — Injury Severity (CatBoost)")
st.caption("ใส่ข้อมูลผู้บาดเจ็บ ระบบจะประเมินระดับความรุนแรงและแนะนำขั้นตอนถัดไป")

# ----------------------------------------------------------
# 📦 Load Model + Encoders + Features
# ----------------------------------------------------------
@st.cache_resource
def load_all():
    try:
        model = joblib.load("predict_catboost_multi.pkl")
        st.success("✅ Loaded: predict_catboost_multi.pkl")
    except:
        model = None
        st.error("❌ ไม่พบ predict_catboost_multi.pkl")

    try:
        encoders = joblib.load("encoders_multi.pkl")
        st.success("✅ Loaded: encoders_multi.pkl")
    except:
        encoders = None
        st.warning("⚠️ ไม่พบ encoders_multi.pkl")

    try:
        with open("features_multi.json", "r") as f:
            features = json.load(f)
        st.success("✅ Loaded: features_multi.json")
    except:
        features = []
        st.warning("⚠️ ไม่พบ features_multi.json")

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
# 🔄 Preprocess Function (Safe Encode)
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

    if "age_group_60plus" not in df.columns:
        df["age_group_60plus"] = (df["age"] >= 60).astype(int)
    if "risk_count" not in df.columns:
        df["risk_count"] = df[["risk1","risk2","risk3","risk4","risk5"]].sum(axis=1)
    if "night_flag" not in df.columns:
        df["night_flag"] = df["is_night"].astype(int)

    df = df.reindex(columns=features, fill_value=0)
    return df

# ----------------------------------------------------------
# 🧠 Run Prediction (CatBoost)
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

# ==========================================================
# 🧩 HOSPITAL AI FOR BUSINESS DASHBOARD (K-Means + Apriori + Summary)
# ==========================================================
st.markdown("---")
st.header("🏥 Hospital AI for Business — Data-Driven Insights")

tab1, tab2, tab3, tab4 = st.tabs([
    "🚑 Real-Time Prediction",
    "📊 Cluster Analysis (K-Means)",
    "🔍 Risk Pattern Mining (Apriori)",
    "📈 Summary Report"
])

# ----------------------------------------------------------
# TAB 1 — CatBoost
# ----------------------------------------------------------
with tab1:
    st.markdown("ระบบประเมินความรุนแรงจากโมเดล **CatBoost** ใช้งานได้ตามปกติ ✅")

# ----------------------------------------------------------
# TAB 2 — K-Means
# ----------------------------------------------------------
with tab2:
    st.subheader("📊 วิเคราะห์กลุ่มผู้บาดเจ็บ (K-Means Clustering)")
    try:
        kmeans = joblib.load("kmeans_cluster_model.pkl")
        scaler = joblib.load("scaler_cluster.pkl")
        st.success("✅ Loaded: kmeans_cluster_model.pkl & scaler_cluster.pkl")
    except:
        st.warning("⚠️ ไม่พบไฟล์โมเดล K-Means หรือ Scaler")
        kmeans, scaler = None, None

    if model and kmeans and scaler and submit:
        X_scaled = scaler.transform(X_input)
        cluster_label = int(kmeans.predict(X_scaled)[0])

        st.markdown(f"### 👥 ผู้บาดเจ็บนี้อยู่ในกลุ่มที่ **{cluster_label}**")
        cluster_desc = {
            0: "ผู้สูงอายุ / ลื่นล้มในบ้าน → เสี่ยงต่ำ",
            1: "วัยทำงาน / ขับรถกลางคืน / เมา → เสี่ยงสูง",
            2: "เด็ก / โรงเรียน / กีฬา → เสี่ยงปานกลาง",
            3: "แรงงาน / ก่อสร้าง → เสี่ยงสูง",
            4: "ทั่วไป / ไม่มีปัจจัยเด่น → เสี่ยงต่ำ"
        }
        st.info(cluster_desc.get(cluster_label, "ยังไม่มีรายละเอียดกลุ่มนี้"))
    else:
        st.info("🕐 เมื่อกรอกข้อมูลและประเมินแล้ว ผลกลุ่มจะปรากฏที่นี่")

# ----------------------------------------------------------
# TAB 3 — Apriori
# ----------------------------------------------------------
with tab3:
    st.subheader("🔍 วิเคราะห์ความสัมพันธ์ของปัจจัยเสี่ยง (Apriori Association Rules)")
    try:
        rules_minor = joblib.load("apriori_rules_minor.pkl")
        rules_severe = joblib.load("apriori_rules_severe.pkl")
        rules_fatal = joblib.load("apriori_rules_fatal.pkl")
        st.success("✅ Loaded Apriori Rules")
    except:
        st.warning("⚠️ ไม่พบไฟล์กฎ Apriori")
        rules_minor = rules_severe = rules_fatal = None

    if model and submit:
        if label == "เสี่ยงน้อย" and rules_minor is not None:
            df_rules = rules_minor.head(5)
        elif label == "เสี่ยงปานกลาง" and rules_severe is not None:
            df_rules = rules_severe.head(5)
        elif label == "เสี่ยงมาก" and rules_fatal is not None:
            df_rules = rules_fatal.head(5)
        else:
            df_rules = pd.DataFrame()

        if not df_rules.empty:
            st.dataframe(df_rules[["antecedents","consequents","support","confidence","lift"]])
        else:
            st.info("📭 ยังไม่มีกฎสำหรับประเภทนี้")

# ----------------------------------------------------------
# TAB 4 — Summary Report (Logging + Dashboard)
# ----------------------------------------------------------
with tab4:
    st.subheader("📈 สรุปผลการประเมินย้อนหลัง (AI Summary Dashboard)")
    if submit and model is not None:
        result_dict = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "age": age, "sex": sex,
            "is_night": int(is_night),
            "head_injury": int(head_injury),
            "mass_casualty": int(mass_casualty),
            "predicted_severity": label,
            "cluster_label": cluster_label if "cluster_label" in locals() else None,
        }
        df_result = pd.DataFrame([result_dict])
        if os.path.exists("results_log.csv"):
            df_result.to_csv("results_log.csv", mode="a", header=False, index=False)
        else:
            df_result.to_csv("results_log.csv", index=False)
        st.success("🧾 บันทึกผลแล้ว (results_log.csv)")

    if os.path.exists("results_log.csv"):
        df_log = pd.read_csv("results_log.csv")
        st.metric("จำนวนเคสทั้งหมด", f"{len(df_log):,}")
        st.bar_chart(df_log["predicted_severity"].value_counts())
    else:
        st.info("ยังไม่มีข้อมูลในระบบ — กรุณาประเมินอย่างน้อย 1 ครั้ง")
st.markdown("---")
st.subheader("🧹 จัดการข้อมูลบันทึก")

if st.button("🗑️ ล้างประวัติทั้งหมด"):
    if os.path.exists("results_log.csv"):
        os.remove("results_log.csv")
        st.success("✅ ล้างข้อมูลบันทึกทั้งหมดเรียบร้อยแล้ว!")
    else:
        st.info("⚠️ ยังไม่มีไฟล์บันทึกในระบบ")
