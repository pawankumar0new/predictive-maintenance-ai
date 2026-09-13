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

# ============================================
# SMART MODEL LOADING
# ============================================

@st.cache_resource
def load_model():
    """Smart model loader that tries multiple paths."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(current_dir, 'models', 'random_forest_model.pkl'),
        os.path.join(os.path.dirname(current_dir), 'models', 'random_forest_model.pkl'),
        os.path.join(current_dir, 'random_forest_model.pkl'),
        os.path.join(current_dir, 'models', 'production', 'random_forest_model.pkl'),
        '/mount/src/predictive-maintenance-ai/models/random_forest_model.pkl',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, path
            except Exception:
                continue
    return None, None

model, model_path = load_model()

# ============================================
# PRESET MACHINE SCENARIOS
# ============================================

# Define preset scenarios for auto-fill
PRESETS = {
    "🟢 Healthy Machine": {
        "air_temp": 298.1,
        "process_temp": 308.6,
        "speed": 1551,
        "torque": 42.8,
        "tool_wear": 0,
        "description": "New machine, all parameters optimal"
    },
    "🟡 Slightly Worn Machine": {
        "air_temp": 300.5,
        "process_temp": 310.2,
        "speed": 1450,
        "torque": 45.5,
        "tool_wear": 120,
        "description": "Moderate tool wear, still operational"
    },
    "🟠 Warning Zone": {
        "air_temp": 302.8,
        "process_temp": 312.5,
        "speed": 1380,
        "torque": 55.2,
        "tool_wear": 175,
        "description": "High tool wear, increased torque"
    },
    "🔴 High Risk - Overheating": {
        "air_temp": 304.5,
        "process_temp": 313.8,
        "speed": 1338,
        "torque": 60.5,
        "tool_wear": 205,
        "description": "Overheating, high wear, near failure"
    },
    "🔴 Critical Failure Zone": {
        "air_temp": 305.0,
        "process_temp": 314.0,
        "speed": 1300,
        "torque": 65.0,
        "tool_wear": 250,
        "description": "Critical condition, immediate action needed"
    },
    "⚡ High Torque Stress": {
        "air_temp": 300.0,
        "process_temp": 310.0,
        "speed": 1200,
        "torque": 75.0,
        "tool_wear": 100,
        "description": "Excessive torque on the machine"
    },
    "🌡️ Temperature Extreme": {
        "air_temp": 315.0,
        "process_temp": 345.0,
        "speed": 1500,
        "torque": 45.0,
        "tool_wear": 100,
        "description": "Temperature outside normal range"
    },
    "⏱️ Tool Wear Critical": {
        "air_temp": 300.0,
        "process_temp": 310.0,
        "speed": 1500,
        "torque": 40.0,
        "tool_wear": 280,
        "description": "Tool wear near maximum limit"
    }
}

# ============================================
# SESSION STATE FOR AUTO-FILL
# ============================================

# Initialize session state for input values
if 'air_temp' not in st.session_state:
    st.session_state.air_temp = 300.0
if 'process_temp' not in st.session_state:
    st.session_state.process_temp = 310.0
if 'speed' not in st.session_state:
    st.session_state.speed = 1500
if 'torque' not in st.session_state:
    st.session_state.torque = 40.0
if 'tool_wear' not in st.session_state:
    st.session_state.tool_wear = 150

def apply_preset(preset_name):
    """Apply a preset scenario to the input fields."""
    preset = PRESETS[preset_name]
    st.session_state.air_temp = float(preset['air_temp'])
    st.session_state.process_temp = float(preset['process_temp'])
    st.session_state.speed = int(preset['speed'])
    st.session_state.torque = float(preset['torque'])
    st.session_state.tool_wear = int(preset['tool_wear'])

def reset_fields():
    """Reset all fields to default values."""
    st.session_state.air_temp = 300.0
    st.session_state.process_temp = 310.0
    st.session_state.speed = 1500
    st.session_state.torque = 40.0
    st.session_state.tool_wear = 150

# ============================================
# SIDEBAR
# ============================================

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
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model not found")
    
    st.markdown("---")
    st.caption("Made for beginner ML demonstration")

# ============================================
# AUTO-FILL SCENARIOS SECTION
# ============================================

st.subheader("🎯 Quick Test Scenarios")
st.markdown("*Click any scenario below to auto-fill the parameters*")

# Create columns for preset buttons
# First row - 3 buttons
row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    if st.button("🟢 Healthy Machine", use_container_width=True, help="New machine, all parameters optimal"):
        apply_preset("🟢 Healthy Machine")
        st.rerun()

with row1_col2:
    if st.button("🟡 Slightly Worn", use_container_width=True, help="Moderate tool wear, still operational"):
        apply_preset("🟡 Slightly Worn Machine")
        st.rerun()

with row1_col3:
    if st.button("🟠 Warning Zone", use_container_width=True, help="High tool wear, increased torque"):
        apply_preset("🟠 Warning Zone")
        st.rerun()

# Second row - 3 buttons
row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    if st.button("🔴 High Risk", use_container_width=True, help="Overheating, high wear, near failure"):
        apply_preset("🔴 High Risk - Overheating")
        st.rerun()

with row2_col2:
    if st.button("🔴 Critical", use_container_width=True, help="Critical condition, immediate action needed"):
        apply_preset("🔴 Critical Failure Zone")
        st.rerun()

with row2_col3:
    if st.button("🔄 Reset Fields", use_container_width=True, help="Reset to default values"):
        reset_fields()
        st.rerun()

# Third row - Special scenarios
row3_col1, row3_col2, row3_col3 = st.columns(3)

with row3_col1:
    if st.button("⚡ High Torque", use_container_width=True, help="Excessive torque on the machine"):
        apply_preset("⚡ High Torque Stress")
        st.rerun()

with row3_col2:
    if st.button("🌡️ Temp Extreme", use_container_width=True, help="Temperature outside normal range"):
        apply_preset("🌡️ Temperature Extreme")
        st.rerun()

with row3_col3:
    if st.button("⏱️ Wear Critical", use_container_width=True, help="Tool wear near maximum limit"):
        apply_preset("⏱️ Tool Wear Critical")
        st.rerun()

st.markdown("---")

# ============================================
# MANUAL INPUT SECTION
# ============================================

st.subheader("📊 Enter Machine Parameters")
st.markdown("*Or modify the auto-filled values below*")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air Temperature [K]", 
        min_value=250.0, 
        max_value=350.0, 
        value=st.session_state.air_temp,
        step=0.5,
        key="input_air_temp"
    )
    st.session_state.air_temp = air_temp
    
    process_temp = st.number_input(
        "Process Temperature [K]", 
        min_value=250.0, 
        max_value=400.0, 
        value=st.session_state.process_temp,
        step=0.5,
        key="input_process_temp"
    )
    st.session_state.process_temp = process_temp
    
    speed = st.number_input(
        "Rotational Speed [rpm]", 
        min_value=1000, 
        max_value=3000, 
        value=st.session_state.speed,
        step=10,
        key="input_speed"
    )
    st.session_state.speed = speed

with col2:
    torque = st.number_input(
        "Torque [Nm]", 
        min_value=0.0, 
        max_value=100.0, 
        value=st.session_state.torque,
        step=0.5,
        key="input_torque"
    )
    st.session_state.torque = torque
    
    tool_wear = st.number_input(
        "Tool Wear [min]", 
        min_value=0, 
        max_value=300, 
        value=st.session_state.tool_wear,
        step=5,
        key="input_tool_wear"
    )
    st.session_state.tool_wear = tool_wear

# ============================================
# SHOW CURRENT SCENARIO INFO
# ============================================

# Detect which preset matches current values (or custom)
current_scenario = "✏️ Custom Values"
for name, preset in PRESETS.items():
    if (abs(preset['air_temp'] - air_temp) < 0.1 and
        abs(preset['process_temp'] - process_temp) < 0.1 and
        preset['speed'] == speed and
        abs(preset['torque'] - torque) < 0.1 and
        preset['tool_wear'] == tool_wear):
        current_scenario = name
        break

st.info(f"**Current Scenario:** {current_scenario}")

# ============================================
# PREDICT BUTTON
# ============================================

if st.button("🔮 Predict Machine Health", type="primary", use_container_width=True):
    if model is None:
        st.error("❌ Model not found. Please train the model first.")
    else:
        try:
            # Preprocess
            temp_diff = process_temp - air_temp
            power = torque * speed * (2 * np.pi / 60)
            wear_torque = tool_wear * torque
            
            features = np.array([[air_temp, process_temp, speed, torque, tool_wear,
                                  temp_diff, power, wear_torque]])
            
            # Predict
            prob = model.predict_proba(features)[0][1]
            pred = model.predict(features)[0]
            
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
                st.metric("Status", f"{emoji} {risk_level}")
            
            with col2:
                st.metric("Failure Risk", f"{failure_probability}%")
            
            with col3:
                st.metric("Health Score", f"{health_score} / 100")
            
            # Health score progress bar
            st.markdown("### Machine Health")
            st.progress(health_score / 100)
            
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
                risk_factors.append(f"⚠️ **Tool wear is high:** {tool_wear} min (threshold: 200)")
            if torque > 60:
                risk_factors.append(f"⚠️ **Torque is high:** {torque} Nm (threshold: 60)")
            if air_temp > 320:
                risk_factors.append(f"⚠️ **Air temperature is high:** {air_temp} K (threshold: 320)")
            if process_temp > 340:
                risk_factors.append(f"⚠️ **Process temperature is high:** {process_temp} K (threshold: 340)")
            if speed > 2500:
                risk_factors.append(f"⚠️ **Speed is high:** {speed} rpm (threshold: 2500)")
            if speed < 1200:
                risk_factors.append(f"⚠️ **Speed is low:** {speed} rpm (threshold: 1200)")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.success("✅ All parameters are within normal ranges.")
            
            # Input summary
            with st.expander("📝 View Input Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Features:**")
                    st.write(f"- Air Temperature: {air_temp} K")
                    st.write(f"- Process Temperature: {process_temp} K")
                    st.write(f"- Rotational Speed: {speed} rpm")
                with col2:
                    st.write("**Additional Info:**")
                    st.write(f"- Torque: {torque} Nm")
                    st.write(f"- Tool Wear: {tool_wear} min")
                    st.write(f"- Temp Difference: {temp_diff:.2f} K")
                
                st.write("**Engineered Features:**")
                st.write(f"- Power: {power:.2f} W")
                st.write(f"- Wear × Torque: {wear_torque:.2f}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.caption("🔧 Smart Machine Health & Early Warning System | Predictive Maintenance AI")