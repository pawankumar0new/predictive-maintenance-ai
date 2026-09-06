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
    st.subheader("📊 Enter Machine Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        air_temp = st.number_input(
            "Air Temperature [K]", 
            min_value=250.0, 
            max_value=350.0, 
            value=300.0,
            step=0.5
        )
        process_temp = st.number_input(
            "Process Temperature [K]", 
            min_value=250.0, 
            max_value=400.0, 
            value=310.0,
            step=0.5
        )
        speed = st.number_input(
            "Rotational Speed [rpm]", 
            min_value=1000, 
            max_value=3000, 
            value=1500,
            step=10
        )
    
    with col2:
        torque = st.number_input(
            "Torque [Nm]", 
            min_value=0.0, 
            max_value=100.0, 
            value=40.0,
            step=0.5
        )
        tool_wear = st.number_input(
            "Tool Wear [min]", 
            min_value=0, 
            max_value=300, 
            value=150,
            step=5
        )
    
    if st.button("🔮 Predict with Multi-Agent System", type="primary"):
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
    
    ### Architecture
    """)

st.markdown("---")
st.caption("🤖 Multi-Agent Predictive Maintenance System | Built with Streamlit")