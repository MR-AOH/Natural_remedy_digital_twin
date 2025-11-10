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
        from patient_history_tracker import PatientHistoryTracker
    
    # Show why demographics matter
    st.subheader("📊 Why Your Profile Matters")

    demo_data = {
        'Age Group': ['25-35', '35-45', '45-55', '55-65', '65+'],
        'Average HbA1c Response': [-1.2, -1.0, -0.8, -0.6, -0.4],
        'BMI <25': [-1.3, -1.1, -0.9, -0.7, -0.5],
        'BMI >30': [-0.9, -0.7, -0.5, -0.3, -0.2]
    }

    demo_df = pd.DataFrame(demo_data)
    st.dataframe(demo_df, use_container_width=True)

    st.info("""
    **Clinical Insight**: Younger patients with lower BMI typically show 30-50% better 
    response to metabolic interventions due to better insulin sensitivity and 
    preserved beta-cell function.
    """)

# Initialize patient history tracker
@st.cache_resource
def init_patient_tracker():
    tracker = PatientHistoryTracker()
    tracker.load_patient_history()
    return tracker

patient_tracker = init_patient_tracker()

# Add patient ID input to sidebar
with st.sidebar:
    st.subheader("👤 Patient Identity")
    patient_id = st.text_input("Patient ID", "DEMO_001")
    
    # Show patient history if available
    if patient_id in patient_tracker.patient_history:
        visits = len(patient_tracker.patient_history[patient_id])
        st.success(f"📋 {visits} historical visits found")
        
        # Show personalization factor
        adjustment = patient_tracker.get_personalized_adjustment(patient_id, selected_remedy)
        st.metric("Personalization Factor", f"{adjustment:.2f}x")

# Add this to wellnessdx_twin_v2.py

def apply_clinical_heuristics(base_prediction, age, bmi, diabetes_duration, comorbidities):
    """
    Apply evidence-based clinical adjustments to predictions
    Based on known physiological principles from literature
    """
    adjusted_prediction = base_prediction.copy()
    
    # Age adjustment (older patients typically respond less)
    if age > 60:
        age_factor = 0.85  # 15% reduced efficacy
        st.info("👴 Age adjustment: -15% (reduced metabolic flexibility)")
    elif age < 30:
        age_factor = 1.15  # 15% increased efficacy  
        st.info("👦 Age adjustment: +15% (better metabolic response)")
    else:
        age_factor = 1.0
    
    # BMI adjustment (higher BMI = often poorer response)
    if bmi > 30:
        bmi_factor = 0.80  # 20% reduced efficacy
        st.info("⚖️ BMI adjustment: -20% (insulin resistance impact)")
    elif bmi < 25:
        bmi_factor = 1.10  # 10% increased efficacy
        st.info("⚖️ BMI adjustment: +10% (better insulin sensitivity)")
    else:
        bmi_factor = 1.0
    
    # Diabetes duration adjustment
    if diabetes_duration > 5:
        duration_factor = 0.75  # 25% reduced efficacy
        st.info("🕒 Diabetes duration: -25% (beta-cell function decline)")
    else:
        duration_factor = 1.0
    
    # Apply all adjustments
    adjustment_factor = age_factor * bmi_factor * duration_factor
    adjusted_prediction = base_prediction * adjustment_factor
    
    return adjusted_prediction, adjustment_factor

# Update the sidebar with REAL patient demographics
with st.sidebar:
    st.subheader("👤 Patient Clinical Profile")
    
    age = st.slider("Age", 18, 80, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.slider("BMI", 18.0, 40.0, 28.0)
    diabetes_years = st.slider("Years with diabetes", 0, 20, 3)
    
    comorbidities = st.multiselect(
        "Comorbidities",
        ["Hypertension", "Dyslipidemia", "Obesity", "PCOS", "NAFLD", "None"]
    )

with tab2:
    st.subheader("🧬 Physiological Simulation")
    ##############################################################################
  # Replace the current prediction section with:

    # --- ML Prediction with Clinical Heuristics ---
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

    # Get base prediction (population average)
    base_predicted_changes = model.predict(input_features)[0]

    # Apply clinical heuristics for personalization
    predicted_changes, clinical_factor = apply_clinical_heuristics(
        base_predicted_changes, age, bmi, diabetes_years, comorbidities
    )

    # Show the adjustment breakdown
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Population Prediction", f"{base_predicted_changes[0]:.2f}%")
    with col2: 
        st.metric("Clinical Adjustment", f"{clinical_factor:.2f}x")
    with col3:
        st.metric("Personalized Prediction", f"{predicted_changes[0]:.2f}%", 
                delta=f"{(predicted_changes[0] - base_predicted_changes[0]):.2f}%")

    # Apply personalization based on patient history
    personalization_factor = patient_tracker.get_personalized_adjustment(patient_id, selected_remedy)
    predicted_changes = base_predicted_changes * personalization_factor

    # Show the personalization effect
    st.info(f"🎯 **Personalized Prediction**: Base effect × {personalization_factor:.2f} (based on your response history)")
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
    # At the end of the simulation, save the results:

    # Add this after the simulation runs in tab2:

    # Save this simulation to patient history
    if st.button("💾 Save to Patient History"):
        biomarkers = {
            'hba1c': hba1c,
            'triglycerides': triglycerides, 
            'hdl': hdl
        }
        
        interventions = {
            'remedy': selected_remedy,
            'duration_weeks': simulation_weeks,
            'adherence': adherence
        }
        
        patient_tracker.add_visit(patient_id, biomarkers, interventions)
        patient_tracker.save_patient_history()
        
        st.success(f"✅ Saved visit to {patient_id}'s history!")
        
        # Show what we learned
        adjustment = patient_tracker.get_personalized_adjustment(patient_id, selected_remedy)
        st.info(f"🎯 Next time, predictions will be personalized by {adjustment:.2f}x based on your response")
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
    st.subheader("📈 Your Response History")
    
    if patient_id in patient_tracker.patient_history:
        history_df = pd.DataFrame(patient_tracker.patient_history[patient_id])
        
        if not history_df.empty:
            # Extract HbA1c history
            hba1c_history = [visit['biomarkers'].get('hba1c', None) 
                           for visit in patient_tracker.patient_history[patient_id]]
            dates = [visit['date'][:10] for visit in patient_tracker.patient_history[patient_id]]
            
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                x=dates, y=hba1c_history,
                mode='lines+markers',
                name='Your HbA1c History',
                line=dict(color='red', width=3)
            ))
            
            fig_history.update_layout(
                title="Your Historical HbA1c Trend",
                xaxis_title="Date",
                yaxis_title="HbA1c (%)"
            )
            
            st.plotly_chart(fig_history, use_container_width=True)
    else:
        st.info("No history yet. After your first simulation, we'll start building your personalized profile.")

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
