import streamlit as st
import joblib
import numpy as np
import time
import base64

# Load the model
model = joblib.load("Classification.joblib")

# Set page configuration
st.set_page_config(page_title="Forest Fire Prediction", page_icon="🔥", layout="wide")

# Background Image Handling
def add_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            background_style = f"""
            <style>
            .stApp {{
                background-image: url("data:image/gif;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                opacity: 0.7;
            }}
            </style>
            """
            st.markdown(background_style, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load background: {e}")

# Add background
try:
    add_background("fire-burn.gif")
except:
    pass

# Title
st.markdown("<h1 style='text-align: center; color: white;'>Forest Fire Prediction System 🔥</h1>", unsafe_allow_html=True)

# Create two columns
left_column, right_column = st.columns(2)

# Use the left column for input
with left_column:
    st.write("Enter the parameters to predict fire risk:")
    
    # User Input
    rain = st.number_input("Rain (mm)", 
                            min_value=0.0, 
                            max_value=50.0, 
                            value=0.0, 
                            step=0.1,
                            key="rain_input")
    
    ffmc = st.number_input("FFMC (Fine Fuel Moisture Code)", 
                            min_value=0.0, 
                            max_value=100.0, 
                            value=0.0, 
                            step=0.1,
                            key="ffmc_input")
    
    dmc = st.number_input("DMC (Duff Moisture Code)", 
                           min_value=0.0, 
                           max_value=300.0, 
                           value=0.0, 
                           step=0.1,
                           key="dmc_input")
    
    isi = st.number_input("ISI (Initial Spread Index)", 
                           min_value=0.0, 
                           max_value=50.0, 
                           value=0.0, 
                           step=0.1,
                           key="isi_input")
    
    # Prediction Button
    predict_button = st.button("Predict Fire Risk")

# Use the right column for results
with right_column:
    # Create a container for results
    result_container = st.container()
    
    # Show prediction if button is clicked
    if predict_button:
        # Validate input
        if rain == 0 and ffmc == 0 and dmc == 0 and isi == 0:
            with result_container:
                st.warning("Please enter valid input values.")
        else:
            # Use st.spinner() as a context manager
            with st.spinner("Analyzing Fire Risk..."):
                # Simulate processing time
                time.sleep(2)
                
                # Prepare input data
                input_data = np.array([[rain, ffmc, dmc, isi]])
                
                # Predict using the model
                prediction = model.predict(input_data)
            
            # Display results in the container
            with result_container:
                # Clear any previous content
                st.empty()
                
                if prediction == 1:
                    st.markdown("<h2 style='color: red;'>🔥 Fire Detected!</h2>", unsafe_allow_html=True)
                    st.markdown("**Fire Prevention Tips:**")
                    # Combine all tips in one markdown block
                    prevention_tips = """
                    - Avoid open flames in dry areas.
                    - Report any smoke or small fires immediately.
                    - Maintain firebreaks around properties.
                    - Follow local fire regulations and alerts.
                    """
                    st.markdown(prevention_tips)
                else:
                    st.markdown("<h2 style='color: green;'>✅ No Fire Risk</h2>", unsafe_allow_html=True)

# Additional styling to improve readability
st.markdown("""
<style>
.stNumberInput > div > div > input {
    color: black;
    background-color: rgba(255, 255, 255, 0.8);
}
.stButton > button {
    width: 100%;
    background-color: orange !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)