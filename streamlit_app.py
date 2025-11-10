# File: wellnessdx_twin_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import plotly.graph_objects as go

# --- Load Resources ---
@st.cache_resource
def load_model():
    return joblib.load("wellnessdx_v3_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("wellnessdx_enhanced_with_sources.csv")

model = load_model()
df = load_data()

st.set_page_config(page_title="WellnessDX Digital Twin v2", layout="wide")
st.title("💊 WellnessDX Digital Twin: Physiological Simulation Engine")

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    simulation_weeks = st.slider("Simulation Duration (weeks)", 4, 24, 12)
    adherence = st.slider("Adherence Level (%)", 0, 100, 85)
    exercise_factor = st.slider("Exercise Boost (hrs/week)", 0.0, 10.0, 3.0)
    
    st.markdown("---")
    st.caption("🔬 Based on methodologies from:")
    st.caption("• Twin Health (Neural Networks)")
    st.caption("• BioTwin (Bayesian Inference)")
    st.caption("• Unlearn.ai (Differential Equations)")

# --- Main Interface ---
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 Simulation", "🔬 Analysis"])

with tab1:
    st.subheader("Step 1: Select Natural Remedy")
    interventions = df['intervention'].unique()
    selected_remedy = st.selectbox("Intervention", interventions)
    
    # Show evidence
    remedy_data = df[df['intervention'] == selected_remedy]
    st.metric("Studies Available", len(remedy_data))
    avg_hba1c = remedy_data['hba1c_change'].mean()
    st.metric("Average HbA1c Change", f"{avg_hba1c:.2f}%", 
              delta=f"{abs(avg_hba1c):.1f}% reduction" if avg_hba1c < 0 else None)
    
    st.subheader("Step 2: Baseline Biomarkers")
    col1, col2, col3 = st.columns(3)
    with col1:
        hba1c = st.number_input("HbA1c (%)", 5.0, 12.0, 7.5)
    with col2:
        triglycerides = st.number_input("Triglycerides (mg/dL)", 50, 500, 180)
    with col3:
        hdl = st.number_input("HDL (mg/dL)", 20, 100, 45)

with tab2:
    st.subheader("🧬 Physiological Simulation")
    
    # --- ML Prediction ---
    input_df = pd.DataFrame({
        'intervention': [selected_remedy],
        'study_type': ['PubMed']
    })
    input_features = pd.get_dummies(input_df)
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_features.columns:
            input_features[col] = 0
    input_features = input_features[model_features]
    
    predicted_changes = model.predict(input_features)[0]
    
    # --- Physiological Model ---
    def metabolic_dynamics(state, t, intervention_effect, params, adherence_factor):
        """
        Differential equations modeling metabolic system
        Based on Twin Health's metabolic flexibility approach
        """
        hba1c, trig, hdl = state
        
        # Physiological parameters (literature-derived)
        k_glucose = params['glucose_clearance']
        k_lipid = params['lipid_metabolism']
        k_hdl_prod = params['hdl_production']
        
        # Intervention effects (scaled by adherence)
        delta_hba1c = intervention_effect['hba1c'] * adherence_factor
        delta_trig = intervention_effect['trig'] * adherence_factor
        delta_hdl = intervention_effect['hdl'] * adherence_factor
        
        # Differential equations
        dhba1c_dt = -k_glucose * (hba1c - 5.0) + delta_hba1c
        dtrig_dt = -k_lipid * (trig - 100) + delta_trig
        dhdl_dt = k_hdl_prod * (60 - hdl) + delta_hdl
        
        return [dhba1c_dt, dtrig_dt, dhdl_dt]
    
    # Simulation parameters
    time_points = np.linspace(0, simulation_weeks, 200)
    initial_state = [hba1c, triglycerides, hdl]
    
    # Intervention effects (distributed over time)
    intervention_effect = {
        'hba1c': predicted_changes[0] / simulation_weeks,
        'trig': predicted_changes[1] / simulation_weeks,
        'hdl': predicted_changes[2] / simulation_weeks
    }
    
    # Physiological parameters
    params = {
        'glucose_clearance': 0.12 + (exercise_factor / 100),
        'lipid_metabolism': 0.08 + (exercise_factor / 80),
        'hdl_production': 0.05 + (exercise_factor / 120)
    }
    
    adherence_factor = adherence / 100
    
    # Solve ODEs
    solution = odeint(metabolic_dynamics, initial_state, time_points, 
                      args=(intervention_effect, params, adherence_factor))
    # Add to tab2 after the ODE simulation
    st.subheader("📊 Uncertainty Quantification")

    # Import the Bayesian module
    from bayesian_module import calculate_prediction_intervals, monte_carlo_simulation

    # Calculate prediction intervals
    literature_std = [0.3, 15.0, 5.0]  # Standard deviations from literature
    intervals = calculate_prediction_intervals(
        predicted_changes, 
        literature_std
    )

    # Display confidence intervals
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "HbA1c Prediction",
            f"{intervals['hba1c']['mean']:.2f}%",
            f"±{intervals['hba1c']['std']:.2f}"
        )
        st.caption(
            f"95% CI: [{intervals['hba1c']['lower']:.2f}, "
            f"{intervals['hba1c']['upper']:.2f}]"
        )

    with col2:
        st.metric(
            "Triglycerides Prediction",
            f"{intervals['triglycerides']['mean']:.0f}",
            f"±{intervals['triglycerides']['std']:.0f}"
        )
        st.caption(
            f"95% CI: [{intervals['triglycerides']['lower']:.0f}, "
            f"{intervals['triglycerides']['upper']:.0f}]"
        )

    with col3:
        st.metric(
            "HDL Prediction",
            f"{intervals['hdl']['mean']:.0f}",
            f"±{intervals['hdl']['std']:.0f}"
        )
        st.caption(
            f"95% CI: [{intervals['hdl']['lower']:.0f}, "
            f"{intervals['hdl']['upper']:.0f}]"
        )

    # Monte Carlo visualization
    monte_carlo_results = monte_carlo_simulation(
        initial_state,
        intervention_effect,
        n_simulations=500
    )

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(
        x=monte_carlo_results[:, 0],
        name='HbA1c Distribution',
        opacity=0.7,
        nbinsx=30
    ))

    fig_mc.update_layout(
        title="Monte Carlo Simulation: HbA1c Outcome Distribution",
        xaxis_title="Final HbA1c (%)",
        yaxis_title="Frequency",
        showlegend=True
    )

    st.plotly_chart(fig_mc, use_container_width=True)



    # --- Visualization ---
    fig = go.Figure()
    
    # HbA1c trajectory
    fig.add_trace(go.Scatter(
        x=time_points, 
        y=solution[:, 0],
        name='HbA1c',
        mode='lines',
        line=dict(color='red', width=3)
    ))
    
    # Normal range shading
    fig.add_hrect(
        y0=4.0, y1=5.7, 
        fillcolor="green", 
        opacity=0.1, 
        annotation_text="Normal Range", 
        annotation_position="top left"
    )
    
    fig.update_layout(
        title=f"HbA1c Trajectory with {selected_remedy}",
        xaxis_title="Weeks",
        yaxis_title="HbA1c (%)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show final predictions
    col1, col2, col3 = st.columns(3)
    final_hba1c = solution[-1, 0]
    final_trig = solution[-1, 1]
    final_hdl = solution[-1, 2]
    
    with col1:
        st.metric(
            "Final HbA1c", 
            f"{final_hba1c:.2f}%", 
            f"{final_hba1c - hba1c:.2f}%"
        )
    with col2:
        st.metric(
            "Final Triglycerides", 
            f"{final_trig:.0f} mg/dL",
            f"{final_trig - triglycerides:.0f}"
        )
    with col3:
        st.metric(
            "Final HDL", 
            f"{final_hdl:.0f} mg/dL",
            f"{final_hdl - hdl:.0f}"
        )

with tab3:
    st.subheader("🔬 Multi-Biomarker Analysis")
    
    # 3D trajectory plot
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=solution[:, 0],
        y=solution[:, 1],
        z=solution[:, 2],
        mode='lines+markers',
        marker=dict(
            size=3,
            color=time_points,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Weeks")
        ),
        line=dict(color='darkblue', width=2),
        text=[f"Week {w:.1f}" for w in time_points],
        hovertemplate='<b>Week %{text}</b><br>' +
                      'HbA1c: %{x:.2f}%<br>' +
                      'Triglycerides: %{y:.0f}<br>' +
                      'HDL: %{z:.0f}<br>'
    )])
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='HbA1c (%)',
            yaxis_title='Triglycerides (mg/dL)',
            zaxis_title='HDL (mg/dL)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title='Metabolic State Space Trajectory',
        height=600
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Evidence table
    st.subheader("📚 Supporting Evidence")
    evidence_cols = [
        'study_id', 
        'title', 
        'publication_year', 
        'hba1c_change', 
        'triglyceride_change', 
        'hdl_change'
    ]
    st.dataframe(
        remedy_data[evidence_cols].head(5), 
        use_container_width=True
    )

# --- Footer ---
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer**: This is a research simulation tool. "
    "Predictions should be validated with healthcare professionals."
)
st.caption(
    f"📊 Model trained on {len(df)} clinical studies. "
    "Accuracy varies by individual metabolic profile."
)
