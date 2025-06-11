"""
Script to run the data validation dashboard.
"""
import os
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_dashboard():
    """Run the Streamlit dashboard."""
    # Get dashboard port from environment or use default
    port = os.getenv('DASHBOARD_PORT', '8501')
    
    # Run Streamlit dashboard
    subprocess.run([
        'streamlit',
        'run',
        'src/data/validation/dashboard.py',
        '--server.port', port,
        '--server.address', '0.0.0.0'
    ])

if __name__ == "__main__":
    run_dashboard() 