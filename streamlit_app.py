import streamlit as st
import sys
import os
import joblib
import numpy as np
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Smart Machine Health & Early Warning System",
    page_icon="🔧",
    layout="centered"
)

st.title("🔧 Smart Machine Health & Early Warning System")
st.markdown("*Predictive Maintenance AI - Multi-Agent System*")
st.markdown("---")

# ============================================
# SMART MODEL LOADING - FINDS THE MODEL ANYWHERE
# ============================================

@st.cache_resource
def load_model():
    """Smart model loader that tries multiple paths."""
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define possible model paths (try these in order)
    possible_paths = [
        # Path 1: Models folder in root
        os.path.join(current_dir, 'models', 'random_forest_model.pkl'),
        
        # Path 2: Models folder one level up (for app/streamlit_app.py)
        os.path.join(os.path.dirname(current_dir), 'models', 'random_forest_model.pkl'),
        
        # Path 3: Current directory
        os.path.join(current_dir, 'random_forest_model.pkl'),
        
        # Path 4: Production folder
        os.path.join(current_dir, 'models', 'production', 'random_forest_model.pkl'),
        
        # Path 5: Staging folder
        os.path.join(current_dir, 'models', 'staging', 'random_forest_model.pkl'),
        
        # Path 6: Absolute path (Streamlit Cloud)
        '/mount/src/predictive-maintenance-ai/models/random_forest_model.pkl',
        
        # Path 7: App directory
        os.path.join(current_dir, 'app', 'models', 'random_forest_model.pkl'),
    ]
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, path
            except Exception as e:
                continue
    
    # If no model found, try to load from GitHub directly (Streamlit Cloud)
    try:
        import urllib.request
        import tempfile
        
        # Try to download from raw GitHub URL
        github_url = "https://raw.githubusercontent.com/pawankumar0new/predictive-maintenance-ai/main/models/random_forest_model.pkl"
        # Note: Raw GitHub URL might not work for binary files
        
        # Alternative: Try with raw.githubusercontent.com
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            urllib.request.urlretrieve(github_url, tmp_file.name)
            model = joblib.load(tmp_file.name)
            os.unlink(tmp_file.name)
            return model, "GitHub (downloaded)"
    except:
        pass
    
    return None, None

# Load model
model, model_path = load_model()

# Show model status in sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    if model is not None:
        st.success(f"✅ Model loaded successfully!")
        st.code(f"📁 {os.path.basename(model_path) if model_path else 'Unknown'}")
        
        # Show model info
        try:
            if hasattr(model, 'n_estimators'):
                st.info(f"🌳 {model.n_estimators} trees in Random Forest")
        except:
            pass
    else:
        st.error("❌ Model not found!")
        st.markdown("""
        **Troubleshooting:**
        1. Make sure `models/random_forest_model.pkl` exists
        2. Check the file is committed to GitHub
        3. Try retraining and recommitting
        """)
    
    st.markdown("---")
    st.caption("Made for beginner ML demonstration")

# ============================================
# MAIN APP CONTENT
# ============================================

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
        st.info("""
        **How to fix this:**
        1. Train the model in your Jupyter notebook
        2. Save it to `models/random_forest_model.pkl`
        3. Commit and push to GitHub
        4. Streamlit will automatically redeploy
        """)
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
                emoji = '✅'
            elif prob < 0.6:
                risk_level = 'Warning'
                emoji = '⚠️'
            else:
                risk_level = 'High Risk'
                emoji = '🚨'
            
            # Display results
            st.markdown("---")
            st.subheader("📋 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Status", 
                    f"{emoji} {risk_level}"
                )
            
            with col2:
                st.metric(
                    "Failure Risk", 
                    f"{failure_probability}%"
                )
            
            with col3:
                st.metric(
                    "Health Score", 
                    f"{health_score} / 100"
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
st.caption("🔧 Smart Machine Health & Early Warning System | Predictive Maintenance AI")