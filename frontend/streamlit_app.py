import streamlit as st
import requests
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Health Dashboard",
    page_icon="🏥",
    layout="wide",
)

# ---------------- HEADING ----------------
st.markdown(
    "<h1 style='text-align:center; font-size:44px; font-weight:800;'>🏥 Health Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- SECTION TITLE ----------------
st.markdown(
    "<h2 style='font-size:26px; font-weight:700;'>🩺 Patient Clinical Information</h2>",
    unsafe_allow_html=True
)

# Row 1 - Name
patient_name = st.text_input("Patient Name")

# Row 2 - 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, step=1)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])

with col3:
    symptom_count = st.number_input("Symptom Count", min_value=0, step=1)

# Symptoms
symptoms = st.text_area(
    "Symptoms (comma separated)",
    placeholder="e.g. chest pain, dizziness, fatigue",
    height=120
)

st.markdown("")

# ---------------- BUTTON ----------------
if st.button("Run Clinical Analysis", use_container_width=True):

    if not patient_name.strip():
        st.warning("Please enter patient name.")
        st.stop()

    if not symptoms.strip():
        st.warning("Please enter symptoms.")
        st.stop()

    payload = {
        "age": age,
        "gender": gender.lower(),
        "symptoms": symptoms,
        "symptom_count": symptom_count,
    }

    progress = st.progress(0)
    for percent in range(0, 101, 25):
        time.sleep(0.2)
        progress.progress(percent)

    try:
        response = requests.post(
            "https://health-prediction-system-j3rs.onrender.com/predict",
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", 0)

            st.markdown("### 🧾 Diagnosis Summary")
            st.write(f"**Patient Name:** {patient_name}")
            st.write(f"**Predicted Condition:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")

        else:
            st.error("API Error")

    except:
        st.error("Backend connection failed")