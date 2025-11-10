# File: bayesian_module.py
import numpy as np
import pandas as pd
from scipy import stats

def calculate_prediction_intervals(predicted_changes, literature_std, confidence=0.95):
    """
    Calculate Bayesian prediction intervals
    
    Args:
        predicted_changes: Array of predicted biomarker changes
        literature_std: Standard deviations from literature
        confidence: Confidence level (default 95%)
    
    Returns:
        Dictionary with lower and upper bounds
    """
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    intervals = {}
    biomarkers = ['hba1c', 'triglycerides', 'hdl']
    
    for i, biomarker in enumerate(biomarkers):
        mean_change = predicted_changes[i]
        std = literature_std[i]
        
        intervals[biomarker] = {
            'mean': mean_change,
            'lower': mean_change - z_score * std,
            'upper': mean_change + z_score * std,
            'std': std
        }
    
    return intervals

def monte_carlo_simulation(initial_state, intervention_effect, n_simulations=1000):
    """
    Monte Carlo simulation for uncertainty quantification
    
    Args:
        initial_state: Initial biomarker values
        intervention_effect: Dict of intervention effects
        n_simulations: Number of Monte Carlo runs
    
    Returns:
        Array of simulated outcomes
    """
    results = []
    
    for _ in range(n_simulations):
        # Add random variation
        noise_hba1c = np.random.normal(0, 0.3)
        noise_trig = np.random.normal(0, 10)
        noise_hdl = np.random.normal(0, 3)
        
        simulated_outcome = [
            initial_state[0] + intervention_effect['hba1c'] + noise_hba1c,
            initial_state[1] + intervention_effect['trig'] + noise_trig,
            initial_state[2] + intervention_effect['hdl'] + noise_hdl
        ]
        
        results.append(simulated_outcome)
    
    return np.array(results)