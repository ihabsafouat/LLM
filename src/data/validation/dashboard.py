"""
Streamlit dashboard for data validation results.
"""
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from .database import DatabaseManager
from .expectations import DataValidator
from .sync_checker import SyncChecker

# Load environment variables
load_dotenv()

# Initialize database manager and sync checker
db = DatabaseManager()
sync_checker = SyncChecker()

def main():
    st.set_page_config(page_title="Data Validation Dashboard", layout="wide")
    
    st.title("Data Validation Dashboard")
    
    # Run synchronization checks
    sync_status = sync_checker.run_checks()
    
    # Display sync status
    st.sidebar.header("System Status")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(
            "Validation Sync",
            "✅" if sync_status['validation_sync'] else "❌",
            help="Checks if validation data is properly synchronized"
        )
    with col2:
        st.metric(
            "Data Quality",
            "✅" if sync_status['data_quality'] else "❌",
            help="Checks for data quality anomalies"
        )
    
    # Sidebar filters
    st.sidebar.header("Filters")
    source = st.sidebar.selectbox(
        "Data Source",
        ["All", "gutenberg", "wikipedia", "arxiv"]
    )
    
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
    )
    
    try:
        # Get validation statistics
        stats = db.get_validation_stats(source if source != "All" else None)
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Runs", stats['total_runs'])
        with col2:
            st.metric("Successful Runs", stats['successful_runs'])
        with col3:
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
            
        # Get latest validation runs
        runs = db.get_latest_validation_runs(limit=100)
        if runs:
            # Convert to DataFrame
            runs_df = pd.DataFrame([
                {
                    'timestamp': run.timestamp,
                    'source': run.source,
                    'success': run.success,
                    'environment': run.environment
                }
                for run in runs
            ])
            
            # Filter by time range
            if time_range != "All time":
                cutoff = datetime.utcnow() - timedelta(
                    days=1 if time_range == "Last 24 hours"
                    else 7 if time_range == "Last 7 days"
                    else 30
                )
                runs_df = runs_df[runs_df['timestamp'] >= cutoff]
                
            # Filter by source
            if source != "All":
                runs_df = runs_df[runs_df['source'] == source]
                
            # Plot validation results over time
            st.subheader("Validation Results Over Time")
            fig = px.scatter(
                runs_df,
                x='timestamp',
                y='source',
                color='success',
                title="Validation Results by Source and Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot success rate by source
            st.subheader("Success Rate by Source")
            success_rates = runs_df.groupby('source')['success'].mean() * 100
            fig = px.bar(
                x=success_rates.index,
                y=success_rates.values,
                title="Success Rate by Source",
                labels={'x': 'Source', 'y': 'Success Rate (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Display quality metrics
        st.subheader("Data Quality Metrics")
        metrics = db.get_quality_metrics(
            source=source if source != "All" else None
        )
        
        if metrics:
            metrics_df = pd.DataFrame([
                {
                    'timestamp': metric.timestamp,
                    'source': metric.source,
                    'metric_name': metric.metric_name,
                    'value': metric.metric_value,
                    'threshold': metric.threshold,
                    'passed': metric.passed
                }
                for metric in metrics
            ])
            
            # Plot metrics over time
            for metric_name in metrics_df['metric_name'].unique():
                metric_data = metrics_df[metrics_df['metric_name'] == metric_name]
                fig = go.Figure()
                
                # Add metric values
                fig.add_trace(go.Scatter(
                    x=metric_data['timestamp'],
                    y=metric_data['value'],
                    name='Value',
                    mode='lines+markers'
                ))
                
                # Add threshold line
                if 'threshold' in metric_data.columns:
                    fig.add_trace(go.Scatter(
                        x=metric_data['timestamp'],
                        y=metric_data['threshold'],
                        name='Threshold',
                        mode='lines',
                        line=dict(dash='dash')
                    ))
                    
                fig.update_layout(
                    title=f"{metric_name} Over Time",
                    xaxis_title="Time",
                    yaxis_title="Value"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        # Display latest validation details
        st.subheader("Latest Validation Details")
        if runs:
            latest_run = runs[0]
            st.write(f"**Source:** {latest_run.source}")
            st.write(f"**Timestamp:** {latest_run.timestamp}")
            st.write(f"**Success:** {latest_run.success}")
            st.write(f"**Environment:** {latest_run.environment}")
            
            if latest_run.statistics:
                st.write("**Statistics:**")
                st.json(latest_run.statistics)
                
            if latest_run.data_docs_url:
                st.write(f"**Data Docs:** [View Report]({latest_run.data_docs_url})")
                
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")
        st.error("Please check the database connection and try again.")

if __name__ == "__main__":
    main() 