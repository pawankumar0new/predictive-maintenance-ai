import streamlit as st
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator_agent import OrchestratorAgent

# Page config
st.set_page_config(
    page_title="Multi-Agent Predictive Maintenance System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Predictive Maintenance System")
st.markdown("*AI Agents working together to predict machine failures*")

# ============================================
# ✅ ADDED: PRESETS AND SESSION STATE
# ============================================

PRESETS = {
    "🟢 Healthy Machine": {
        "air_temp": 298.1, "process_temp": 308.6,
        "speed": 1551, "torque": 42.8, "tool_wear": 0,
        "description": "New machine, all parameters optimal"
    },
    "🟡 Slightly Worn Machine": {
        "air_temp": 300.5, "process_temp": 310.2,
        "speed": 1450, "torque": 45.5, "tool_wear": 120,
        "description": "Moderate tool wear, still operational"
    },
    "🟠 Warning Zone": {
        "air_temp": 302.8, "process_temp": 312.5,
        "speed": 1380, "torque": 55.2, "tool_wear": 175,
        "description": "High tool wear, increased torque"
    },
    "🔴 High Risk - Overheating": {
        "air_temp": 304.5, "process_temp": 313.8,
        "speed": 1338, "torque": 60.5, "tool_wear": 205,
        "description": "Overheating, high wear, near failure"
    },
    "🔴 Critical Failure Zone": {
        "air_temp": 305.0, "process_temp": 314.0,
        "speed": 1300, "torque": 65.0, "tool_wear": 250,
        "description": "Critical condition, immediate action needed"
    },
    "⚡ High Torque Stress": {
        "air_temp": 300.0, "process_temp": 310.0,
        "speed": 1200, "torque": 75.0, "tool_wear": 100,
        "description": "Excessive torque on the machine"
    },
    "🌡️ Temperature Extreme": {
        "air_temp": 315.0, "process_temp": 345.0,
        "speed": 1500, "torque": 45.0, "tool_wear": 100,
        "description": "Temperature outside normal range"
    },
    "⏱️ Tool Wear Critical": {
        "air_temp": 300.0, "process_temp": 310.0,
        "speed": 1500, "torque": 40.0, "tool_wear": 280,
        "description": "Tool wear near maximum limit"
    }
}

# Initialize session state
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
# END OF ADDED CODE
# ============================================

# Initialize orchestrator
@st.cache_resource
def get_orchestrator():
    return OrchestratorAgent()

orchestrator = get_orchestrator()

# Sidebar - Agent Status
with st.sidebar:
    st.header("📊 Agent Status")
    
    if st.button("🔄 Refresh Agent Status"):
        st.rerun()
    
    status = orchestrator.get_agent_status()
    
    for name, agent_status in status.items():
        with st.expander(f"🤖 {name}", expanded=False):
            st.write(f"**Role:** {agent_status['role']}")
            st.write(f"**Status:** {agent_status['status']}")
            st.write(f"**Last Activity:** {agent_status['last_activity']}")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Make Prediction", 
    "📈 History", 
    "🔍 Agent Debug",
    "ℹ️ About"
])

with tab1:
    # ============================================
    # ✅ ADDED: AUTO-FILL SCENARIOS SECTION
    # ============================================
    
    st.subheader("🎯 Quick Test Scenarios")
    st.markdown("*Click any scenario below to auto-fill the parameters*")
    
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
    # END OF AUTO-FILL SECTION
    # ============================================
    
    st.subheader("📊 Enter Machine Parameters")
    st.markdown("*Or modify the auto-filled values below*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ✅ UPDATED: Use session state values
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
        # ✅ UPDATED: Use session state values
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
    # ✅ ADDED: SHOW CURRENT SCENARIO
    # ============================================
    
    # Detect which preset matches current values
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
    # END OF CURRENT SCENARIO
    # ============================================
    
    if st.button("🔮 Predict with Multi-Agent System", type="primary", use_container_width=True):
        with st.spinner("🤖 Agents are processing your request..."):
            features = {
                'air_temp': air_temp,
                'process_temp': process_temp,
                'speed': speed,
                'torque': torque,
                'tool_wear': tool_wear
            }
            
            result = orchestrator.process_prediction_request(features)
            
            if 'error' in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                # Display results
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                pred = result['prediction']
                exp = result['explanation']
                
                # Risk level colors
                risk_colors = {
                    'Normal': '🟢',
                    'Warning': '🟡',
                    'High Risk': '🔴'
                }
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Status", 
                        f"{risk_colors.get(exp['risk_level'], '⚪')} {exp['risk_level']}"
                    )
                
                with col2:
                    st.metric(
                        "Failure Risk", 
                        f"{pred['failure_probability']}%"
                    )
                
                with col3:
                    st.metric(
                        "Health Score", 
                        f"{exp['health_score']} / 100"
                    )
                
                with col4:
                    st.metric(
                        "Confidence", 
                        f"{pred['confidence']}%"
                    )
                
                # ✅ ADDED: Health score progress bar
                st.markdown("### Machine Health")
                st.progress(exp['health_score'] / 100)
                
                # Recommendation
                st.markdown("---")
                st.subheader("🛠️ Recommendation")
                
                if exp['risk_level'] == 'Normal':
                    st.success(exp['message'])
                elif exp['risk_level'] == 'Warning':
                    st.warning(exp['message'])
                else:
                    st.error(exp['message'])
                
                # Risk factors
                if exp.get('risk_factors'):
                    st.subheader("⚠️ Risk Factors")
                    for factor in exp['risk_factors']:
                        st.write(f"- {factor}")
                
                # Show which agents were involved
                with st.expander("🤖 Agent Collaboration Details"):
                    st.write("**Agents Involved:**")
                    st.write("1. **Data Agent** - Validated and preprocessed input")
                    st.write("2. **Prediction Agent** - Made the prediction")
                    st.write("3. **Explanation Agent** - Generated recommendation")
                    st.write(f"\n**Processed Features:** {result['processed_features']}")

with tab2:
    st.subheader("📈 Prediction History")
    
    history = orchestrator.history
    
    if not history:
        st.info("No predictions made yet.")
    else:
        df = pd.DataFrame([{
            'Time': h.get('timestamp', ''),
            'Risk Level': h['explanation']['risk_level'],
            'Health Score': h['explanation']['health_score'],
            'Failure Probability': h['prediction']['failure_probability'],
            'Confidence': h['prediction']['confidence']
        } for h in history])
        
        st.dataframe(df, use_container_width=True)

with tab3:
    st.subheader("🔍 Agent Debug Information")
    
    status = orchestrator.get_agent_status()
    
    for name, agent_status in status.items():
        with st.expander(f"🤖 {name}", expanded=True):
            st.json(agent_status)

with tab4:
    st.subheader("ℹ️ About the Multi-Agent System")
    
    st.markdown("""
    ### How it Works
    
    This system uses **multiple AI agents** working together:
    
    1. **🤖 Data Agent** - Validates and preprocesses input data
    2. **🤖 Prediction Agent** - Makes the actual prediction
    3. **🤖 Explanation Agent** - Generates recommendations and explanations
    4. **🤖 Orchestrator Agent** - Coordinates all agents
    
    ### Benefits of Multi-Agent Approach
    
    - **Modular**: Each agent has a single responsibility
    - **Scalable**: Easy to add new agents
    - **Explainable**: Each step is traceable
    - **Resilient**: Agents can fail independently
    
    ### Quick Test Scenarios
    
    Use the **Quick Test Scenarios** buttons in the Predict tab to instantly 
    test different machine conditions:
    
    - 🟢 **Healthy Machine** - New machine, optimal parameters
    - 🟡 **Slightly Worn** - Moderate tool wear
    - 🟠 **Warning Zone** - High wear, increased torque
    - 🔴 **High Risk** - Overheating, near failure
    - 🔴 **Critical** - Immediate action needed
    
    ### Architecture
    """)

st.markdown("---")
st.caption("🤖 Multi-Agent Predictive Maintenance System | Built with Streamlit")