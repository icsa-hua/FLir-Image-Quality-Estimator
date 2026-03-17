import streamlit as st
from drift_reports.base_drift_report import Report

class TextDriftReport(Report):
    def __init__(self, refe_data, test_data):
        self.refe_data = refe_data
        self.test_data = test_data

    def generate_report(self):
        st.write("To be implemented")
