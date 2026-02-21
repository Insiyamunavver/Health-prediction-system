import streamlit as st
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Healthcare Prediction System",
    page_icon="🏥",
    layout="centered",
)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- TITLE ----------------
st.title("🏥 Healthcare Prediction System")
st.markdown("AI-powered disease prediction using patient symptoms")
st.divider()

# ---------------- INPUT FORM ----------------
st.subheader("📝 Enter Patient Details")

age = st.number_input("Age", min_value=0, max_value=120, step=1)

gender = st.selectbox(
    "Gender",
    ["male", "female"],  # keep lowercase if API expects lowercase
)

symptoms = st.text_area(
    "Symptoms (comma separated)",
    placeholder="e.g. chest pain, dizziness, fatigue",
)

symptom_count = st.number_input(
    "Symptom Count",
    min_value=0,
    step=1,
)

# ---------------- BUTTON ----------------
if st.button("🔍 Predict", use_container_width=True):

    # -------- Validation --------
    if not symptoms.strip():
        st.warning("⚠ Please enter symptoms before predicting.")
        st.stop()

    payload = {
        "age": age,
        "gender": gender,
        "symptoms": symptoms,
        "symptom_count": symptom_count,
    }

    try:
        with st.spinner("🤖 Analyzing symptoms..."):

            response = requests.post(
                "https://health-prediction-system-j3rs.onrender.com/predict",
                json=payload,
                timeout=5,
            )

        # -------- Success --------
        if response.status_code == 200:

            result = response.json()
            prediction = result.get("prediction", "Unknown")

            st.success("✅ Prediction Successful")

            st.metric(
                label="Predicted Disease",
                value=prediction,
            )

            # Save history
            st.session_state.history.append(
                {
                    "age": age,
                    "gender": gender,
                    "symptoms": symptoms,
                    "prediction": prediction,
                }
            )

        # -------- API Error --------
        else:
            st.error(f"🚨 API Error: {response.status_code}")
            st.write(response.text)

    # -------- Connection Error --------
    except requests.exceptions.RequestException as exc:
        st.error("🚨 Could not connect to API")
        st.write(str(exc))

st.divider()

# ---------------- HISTORY ----------------
if st.session_state.history:
    st.subheader("📊 Prediction History")

    for i, entry in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Prediction {i}"):

            st.write(f"**Age:** {entry['age']}")
            st.write(f"**Gender:** {entry['gender']}")
            st.write(f"**Symptoms:** {entry['symptoms']}")
            st.write(f"**Prediction:** {entry['prediction']}")

# ---------------- SIDEBAR ----------------
st.sidebar.title("ℹ About")
st.sidebar.info(
    """
    This AI system predicts diseases based on:
    - Age
    - Gender
    - Symptoms
    - Symptom Count

    Built with:
    - FastAPI 🚀
    - Streamlit 🎨
    - Machine Learning 🤖
    """
)
