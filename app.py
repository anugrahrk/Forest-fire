import streamlit as st
import joblib
import numpy as np
import time
import base64

model = joblib.load("Classification1.joblib")
model2=joblib.load("Regression.joblib")

st.set_page_config(page_title="Forest Fire Prediction", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Project AGNI 🔥</h1>", unsafe_allow_html=True)
st.write("Enter the parameters to predict fire risk:")
rain = st.slider("Rain (mm)", 
                        min_value=0.0, 
                        max_value=50.0, 
                        value=0.0, 
                        step=0.1,
                        key="rain_input")

ffmc = st.slider("FFMC (Fine Fuel Moisture Code)", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=0.0, 
                        step=0.1,
                        key="ffmc_input")

dmc = st.slider("DMC (Duff Moisture Code)", 
                        min_value=0.0, 
                        max_value=300.0, 
                        value=0.0, 
                        step=0.1,
                        key="dmc_input")

isi = st.slider("ISI (Initial Spread Index)", 
                        min_value=0.0, 
                        max_value=50.0, 
                        value=0.0, 
                        step=0.1,
                        key="isi_input")
predict_button = st.button("Predict Fire Risk")
result_container = st.container()
if predict_button:
        with st.spinner("Analyzing Fire Risk..."):
            time.sleep(2)
            input_data = np.array([[rain, ffmc, dmc, isi]])
            prediction = model.predict(input_data)
            prediction2=model2.predict(input_data)
            value=prediction2[0]*3.5
        with result_container:
            st.empty()
            
            if prediction == 1 and value>=50:
                st.markdown("<h2 style='color: red;'>🔥 Fire Detected!</h2>", unsafe_allow_html=True)
                # st.markdown("**Fire Prevention Tips:**")
                # prevention_tips = """
                # - Avoid open flames in dry areas.
                # - Report any smoke or small fires immediately.
                # - Maintain firebreaks around properties.
                # - Follow local fire regulations and alerts.
                # """
                # st.markdown(prevention_tips)
            else:
                st.markdown("<h2 style='color: green;'>✅ No Fire Risk</h2>", unsafe_allow_html=True)

            chance_of_fire = float(value)
        if chance_of_fire < 50:
            st.markdown(f"<h3 style='color: green;'>Chance of Fire: {chance_of_fire:.2f}%</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: red;'>Chance of Fire: {chance_of_fire:.2f}%</h3>", unsafe_allow_html=True)