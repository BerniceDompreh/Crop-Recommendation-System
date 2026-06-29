import streamlit as st
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="🌾 Ghana Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f5fff5;
}

h1{
    color:#2E7D32;
    text-align:center;
}

.stButton>button{
    background-color:#2E7D32;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
}

.stButton>button:hover{
    background-color:#1B5E20;
}

.result{
    background:#E8F5E9;
    padding:20px;
    border-radius:12px;
    border-left:8px solid green;
    font-size:22px;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("crop_prediction_model.pkl")

# -----------------------------
# HEADER
# -----------------------------
st.title("🌾 Smart Crop Recommendation System")

st.image(
    "https://images.unsplash.com/photo-1500937386664-56d1dfef3854?w=1200",
    use_container_width=True
)

st.write("""
### Welcome!

This intelligent crop recommendation system predicts the most suitable crop
based on soil properties and environmental conditions.

Enter the values below and click **Recommend Crop**.
""")

# -----------------------------
# INPUT SECTION
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
    phosphorus = st.number_input("Phosphorus (P)", 0.0, 200.0, 40.0)
    potassium = st.number_input("Potassium (K)", 0.0, 300.0, 40.0)
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, 200.0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🌱 Recommend Crop"):

    input_data = np.array([[nitrogen,
                            phosphorus,
                            potassium,
                            temperature,
                            humidity,
                            ph,
                            rainfall]])

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div class="result">
        🌾 <b>Recommended Crop:</b> {prediction}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("Recommendation generated successfully!")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("ℹ About")

st.sidebar.info("""
This application uses a trained Machine Learning model to recommend the best crop based on:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

Developed using **Streamlit** and **Scikit-Learn**.
""")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
"""
<div class="footer">
Smart Crop Recommendation System 🌱 <br>
Designed for Sustainable Agriculture
</div>
""",
unsafe_allow_html=True)
