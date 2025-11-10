# Add this new module to track patient history
# File: patient_history_tracker.py

import pandas as pd
import json
from datetime import datetime, timedelta
import streamlit as st
class PatientHistoryTracker:
    """Tracks individual patient history for personalized predictions"""
    
    def __init__(self):
        self.patient_history = {}
        
    def add_visit(self, patient_id, biomarkers, interventions, date=None):
        """Add a patient visit with biomarker data"""
        if patient_id not in self.patient_history:
            self.patient_history[patient_id] = []
            
        visit_data = {
            'date': date or datetime.now().isoformat(),
            'biomarkers': biomarkers.copy(),
            'interventions': interventions.copy(),
            'response_rates': {}
        }
        
        # Calculate response rates if we have previous data
        if len(self.patient_history[patient_id]) > 0:
            prev_visit = self.patient_history[patient_id][-1]
            visit_data['response_rates'] = self._calculate_response_rates(
                prev_visit['biomarkers'], biomarkers
            )
        
        self.patient_history[patient_id].append(visit_data)
        
    def _calculate_response_rates(self, prev_biomarkers, current_biomarkers):
        """Calculate how this patient responds to interventions"""
        rates = {}
        for key in prev_biomarkers:
            if key in current_biomarkers:
                change = current_biomarkers[key] - prev_biomarkers[key]
                rates[key] = {
                    'absolute_change': change,
                    'percent_change': (change / prev_biomarkers[key]) * 100
                }
        return rates
    
    def get_personalized_adjustment(self, patient_id, intervention):
        """Get personalized adjustment factors based on patient history"""
        if patient_id not in self.patient_history or len(self.patient_history[patient_id]) < 2:
            return 1.0  # Default adjustment if no history
            
        history = self.patient_history[patient_id]
        recent_response = history[-1].get('response_rates', {})
        
        # Calculate personalization factor based on historical responses
        if 'hba1c' in recent_response:
            # If patient historically responds well, boost prediction
            hba1c_response = abs(recent_response['hba1c']['percent_change'])
            personalization_factor = 1.0 + (hba1c_response / 100)  # 5-20% adjustment
            return min(max(personalization_factor, 0.8), 1.2)  # Bound between 0.8-1.2
        
        return 1.0
    
    def save_patient_history(self, filename="patient_history.json"):
        """Save patient history to file"""
        with open(filename, 'w') as f:
            json.dump(self.patient_history, f, indent=2)
    
    def load_patient_history(self, filename="patient_history.json"):
        """Load patient history from file"""
        try:
            with open(filename, 'r') as f:
                self.patient_history = json.load(f)
        except FileNotFoundError:
            self.patient_history = {}

# Update the main Streamlit app to include patient history
# Add this to wellnessdx_twin_v2.py at the top:

