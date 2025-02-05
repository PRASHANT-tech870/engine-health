import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open('rf_model_gpu.pkl', 'rb') as file:
    model = pickle.load(file)

# Sensor fields
fields = [
    "Cycle", "OpSet1", "OpSet2", "OpSet3", 
    "Primary Temperature Reading", "Secondary Temperature Reading", "Tertiary Temperature Reading", 
    "Quaternary Temperature Reading", "Primary Pressure Reading", "Secondary Pressure Reading", 
    "Tertiary Pressure Reading", "Quaternary Pressure Reading", "Primary Speed Reading", 
    "Secondary Speed Reading", "Tertiary Speed Reading", "Quaternary Speed Reading", 
    "Primary Vibration Reading", "Secondary Vibration Reading", "Primary Flow Reading", 
    "Secondary Flow Reading", "Tertiary Flow Reading", "Pressure Ratio", 
    "Efficiency Indicator", "Power Setting", "Fuel Flow Rate"
]

# Predefined values for GOOD, MODERATE, VERY BAD
predefined_values = {
    "GOOD": [124,42.0046,0.84,100.0,445.0,549.89,1348.37,1110.2,3.91,5.69,137.2,2211.88,8310.53,1.01,41.58,129.39,2388.01,8078.3,9.3771,0.02,329,2212,100.0,10.64,6.3304],
    "MODERATE": [160,0.0009,0.0,100.0,518.67,642.09,1581.5,1393.82,14.62,21.56,552.93,2387.95,9052.64,1.3,47.17,521.69,2387.95,8137.17,8.3756,0.03,391,2388,100.0,38.97,23.3432],
    "VERY BAD": [252,0.002,0.0017,100.0,518.67,642.85,1595.78,1410.08,14.62,21.57,562.77,2388.24,9097.79,1.31,47.74,529.38,2388.22,8169.2,8.279,0.03,393,2388,100.0,39.22,23.6432]
}

# Streamlit UI
st.title("Engine Health Predictor")

# Autofill buttons
for label in predefined_values:
    if st.button(f"Autofill for {label}"):
        for i, field in enumerate(fields):
            st.session_state[field] = predefined_values[label][i]

# Text input for pasting comma-separated values
user_input = st.text_area("Paste comma-separated values", "")

# Apply button for user input
if st.button("Apply"):
    if user_input:
        input_values = list(map(float, user_input.split(',')))
        
        if len(input_values) == len(fields):
            for i, field in enumerate(fields):
                st.session_state[field] = input_values[i]
        else:
            st.error("Invalid input. Ensure the number of values matches the required fields.")

# User input fields
input_data = []
for field in fields:
    value = st.number_input(field, key=field, value=st.session_state.get(field, 0.0))
    input_data.append(value)

# Convert input_data to a pandas DataFrame with dtype float32
input_df = pd.DataFrame([input_data], columns=fields)
input_df = input_df.astype('float32')

# Submit button
if st.button("Submit"):
    prediction = model.predict(input_df)[0]  # Pass DataFrame to model
    result_map = {0: "GOOD", 1: "MODERATE", 2: "VERY BAD"}
    st.success(f"Predicted Condition: {result_map.get(prediction, 'Unknown')}")
