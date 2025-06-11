"""
Script to run the Streamlit dashboard.
"""
import os
from dotenv import load_dotenv
import streamlit as st
from src.data.validation.dashboard import main as dashboard_main

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Set Streamlit configuration
    st.set_page_config(
        page_title="Data Validation Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Run the dashboard
    dashboard_main() 