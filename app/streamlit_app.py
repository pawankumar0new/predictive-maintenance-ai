import streamlit as st
import sys
import os
import joblib
import numpy as np

# Set page config
st.set_page_config(
    page_title="Smart Machine Health & Early Warning System",
    page_icon="🔧",
    layout="centered"
)

st.title("🔧 Smart Machine Health & Early Warning System")
st.markdown("*Predictive Maintenance AI - Multi-Agent System*")
st.markdown("---")

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Model paths - try multiple locations
model_paths = [
    os.path.join(project_root, 'models', 'random_forest_model.pkl'),
    os.path.join(project_root, 'models', 'production', 'random_forest_model.pkl'),
    os.path.join(current_dir, '..', 'models', 'random_forest_model.pkl'),
    'models/random_forest_model.pkl'
]

def load_model():
    """Load the model from various possible locations."""
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, path
            except:
                continue
    return None, None

# Load model with caching
@st.cache_resource
def get_model():
    return load_model()

model, model_path = get_model()

# Sidebar
with st.sidebar:
    st.header("📊 About")
    st.markdown("""
    This system uses a **Random Forest** model trained on the AI4I 2020 dataset.
    
    **Features used:**
    - Air Temperature
    - Process Temperature  
    - Rotational Speed
    - Torque
    - Tool Wear
    
    **Output:**
    - Failure Probability
    - Health Score
    - Risk Level
    - Maintenance Recommendation
    """)
    
    if model:
        st.success(f"✅ Model loaded: {os.path.basename(model_path)}")
    else:
        st.error("❌ Model not found")
    
    st.markdown("---")
    st.caption("Made for beginner ML demonstration")

# Input section
st.subheader("📊 Enter Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air Temperature [K]", 
        min_value=250.0, 
        max_value=350.0, 
        value=300.0,
        step=0.5,
        key="air_temp"
    )
    process_temp = st.number_input(
        "Process Temperature [K]", 
        min_value=250.0, 
        max_value=400.0, 
        value=310.0,
        step=0.5,
        key="process_temp"
    )
    speed = st.number_input(
        "Rotational Speed [rpm]", 
        min_value=1000, 
        max_value=3000, 
        value=1500,
        step=10,
        key="speed"
    )

with col2:
    torque = st.number_input(
        "Torque [Nm]", 
        min_value=0.0, 
        max_value=100.0, 
        value=40.0,
        step=0.5,
        key="torque"
    )
    tool_wear = st.number_input(
        "Tool Wear [min]", 
        min_value=0, 
        max_value=300, 
        value=150,
        step=5,
        key="tool_wear"
    )

# Feature engineering function
def preprocess_features(air_temp, process_temp, speed, torque, tool_wear):
    temp_diff = process_temp - air_temp
    power = torque * speed * (2 * np.pi / 60)
    wear_torque = tool_wear * torque
    return np.array([[air_temp, process_temp, speed, torque, tool_wear, 
                      temp_diff, power, wear_torque]])

# Predict button
if st.button("🔮 Predict", type="primary"):
    if model is None:
        st.error("❌ Model not found. Please train the model first.")
    else:
        try:
            # Preprocess
            features = preprocess_features(
                air_temp, process_temp, speed, torque, tool_wear
            )
            
            # Predict
            prob = model.predict_proba(features)[0][1]
            pred = model.predict(features)[0]
            
            # Calculate metrics
            failure_probability = round(prob * 100, 2)
            health_score = int((1 - prob) * 100)
            
            # Risk level
            if prob < 0.3:
                risk_level = 'Normal'
                risk_color = 'green'
                emoji = '✅'
            elif prob < 0.6:
                risk_level = 'Warning'
                risk_color = 'orange'
                emoji = '⚠️'
            else:
                risk_level = 'High Risk'
                risk_color = 'red'
                emoji = '🚨'
            
            # Display results
            st.markdown("---")
            st.subheader("📋 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Status", 
                    f"{emoji} {risk_level}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Failure Risk", 
                    f"{failure_probability}%",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Health Score", 
                    f"{health_score} / 100",
                    delta=None
                )
            
            # Maintenance recommendation
            st.markdown("---")
            st.subheader("🛠️ Recommendation")
            
            if risk_level == 'Normal':
                st.success("✅ Machine is operating normally. Continue monitoring.")
            elif risk_level == 'Warning':
                st.warning("⚠️ Schedule inspection soon. Monitor key parameters closely.")
            else:
                st.error("🚨 High risk detected! Consider immediate maintenance action.")
            
            # Risk factors
            st.markdown("---")
            st.subheader("📊 Key Risk Factors")
            
            risk_factors = []
            if tool_wear > 200:
                risk_factors.append(f"⚠️ Tool wear is high: {tool_wear} min")
            if torque > 60:
                risk_factors.append(f"⚠️ Torque is high: {torque} Nm")
            if air_temp > 320:
                risk_factors.append(f"⚠️ Air temperature is high: {air_temp} K")
            if process_temp > 340:
                risk_factors.append(f"⚠️ Process temperature is high: {process_temp} K")
            if speed > 2500:
                risk_factors.append(f"⚠️ Speed is high: {speed} rpm")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.write("✅ All parameters are within normal ranges.")
            
            # Input summary
            with st.expander("📝 View Input Parameters"):
                st.write({
                    "Air Temperature": f"{air_temp} K",
                    "Process Temperature": f"{process_temp} K", 
                    "Rotational Speed": f"{speed} rpm",
                    "Torque": f"{torque} Nm",
                    "Tool Wear": f"{tool_wear} min"
                })
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Make sure the model has been trained and saved.")

st.markdown("---")
st.caption("This is a demonstration system. Recommendations are for educational purposes only.")