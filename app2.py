import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import base64

# Load the trained model
baseline_model = joblib.load('flood_model.joblib')  

def preprocess_and_train(data):
    # Convert datetime feature to datetime
    data["DateTime"] = pd.to_datetime(data["DateTime"], format="mixed", dayfirst=True)
    
    # Forward and backward fill missing values
    data = data.ffill(axis=0).bfill(axis=0)
    
    # Encode categorical variables
    data["Season"] = data["Season"].apply(
        lambda x: 1 if x == "Dormant Season" else (0 if x == "Growing Season" else None)
    )
    data["AntecedentRainCondition"] = data["AntecedentRainCondition"].apply(
        lambda x: 0 if x == "AMC I (Dry)" else (1 if x == "AMC II (Average)" else 2)
    )
    
    # Add month column
    data["month"] = data["DateTime"].dt.month
    
    # Set DateTime as index
    data = data.set_index("DateTime")
    
    # Reorder columns
    data = data[[
        "Rain_in", "Season", "AntecedentRain_in", "AntecedentRainCondition", 
        "RainIntensity_in_hr", "PeakRunoff", "TimeToPeak", "month"
    ]]
    
    # Make predictions
    predictions = baseline_model.predict(data)
    data["ChestnutCreek_ft"] = predictions
    
    return data

def train(data):
    # Convert datetime feature to datetime
    data["DateTime"] = pd.to_datetime(data["DateTime"], format="mixed", dayfirst=True)
    # Add month column
    data["month"] = data["DateTime"].dt.month
    
    # Set DateTime as index
    data = data.set_index("DateTime")
    
    # Reorder columns
    data = data[[
        "Rain_in", "Season", "AntecedentRain_in", "AntecedentRainCondition", 
        "RainIntensity_in_hr", "PeakRunoff", "TimeToPeak", "month"
    ]]
    
    # Make predictions
    predictions = baseline_model.predict(data)
    data["ChestnutCreek_ft"] = predictions
    return data

logo_path = "logo_svg.svg"

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo_path, width=300, use_container_width= True)
st.markdown('<div style="text-align: center; font-size: 28px;">River Level Prediction App</div>', unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.write("---") 
# Option 1: Upload a CSV file
st.subheader("Upload a CSV File", divider="gray")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    result = preprocess_and_train(data)
    st.write("Predictions:")
    st.dataframe(result)

    # Download the result as CSV
    csv = result.to_csv(index=True)
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

# Option 2: Manual Input
st.subheader("Manual Input", divider="gray")
with st.form("manual_input"):
    date_time = st.text_input("DateTime (YYYY-MM-DD HH:MM:SS)")
    rain_in = st.number_input("Rain_in", min_value=0.00, format="%.4f")
    season = st.selectbox("Season", ["Dormant Season (1)", "Growing Season (0)"])
    antecedent_rain_in = st.number_input("AntecedentRain_in", min_value=0.00, format="%.4f")
    antecedent_rain_condition = st.selectbox("AntecedentRainCondition", ["AMC I- Dry(0)", "AMC II- Average(1)", "AMC III- Wet(2)"])
    rain_intensity_in_hr = st.number_input("RainIntensity_in_hr", min_value=0.00, format="%.4f")
    peak_runoff = st.number_input("PeakRunoff", min_value=0.00, format="%.4f")
    time_to_peak = st.number_input("TimeToPeak", min_value=0.00, format="%.4f")
    month = pd.to_datetime(date_time).month if date_time else 1
    
    submit_button = st.form_submit_button("Submit")
    
    if submit_button and date_time:
        input_data = pd.DataFrame([{
            "DateTime": date_time,
            "Rain_in": rain_in,
            "Season": 1 if season == "Dormant Season" else 0,
            "AntecedentRain_in": antecedent_rain_in,
            "AntecedentRainCondition": 0 if antecedent_rain_condition == "AMC I (Dry)" else (1 if antecedent_rain_condition == "AMC II (Average)" else 2),
            "RainIntensity_in_hr": rain_intensity_in_hr,
            "PeakRunoff": peak_runoff,
            "TimeToPeak": time_to_peak,
            "month": month
        }])
        
        result = train(input_data)
        st.write("Prediction for the given inputs:")
        st.dataframe(result)
