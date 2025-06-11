"""
Data synchronization checker for validation results.
"""
import os
import logging
from datetime import datetime, timedelta
from sqlalchemy import text
from .database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', '/opt/airflow/logs/sync.log')),
        logging.StreamHandler()
    ]
)

class SyncChecker:
    def __init__(self):
        """Initialize synchronization checker."""
        self.db = DatabaseManager()
        
    def check_validation_sync(self, time_window_minutes=5):
        """Check if validation results are properly synchronized."""
        try:
            # Get latest validation runs
            latest_runs = self.db.get_latest_validation_runs(limit=100)
            if not latest_runs:
                logging.warning("No validation runs found")
                return False
                
            # Check if runs are recent
            latest_timestamp = latest_runs[0].timestamp
            time_diff = datetime.utcnow() - latest_timestamp
            
            if time_diff > timedelta(minutes=time_window_minutes):
                logging.warning(f"Latest validation run is too old: {time_diff}")
                return False
                
            # Check data consistency
            for run in latest_runs:
                # Verify run has associated expectations
                expectations = self.db.Session().query(text("""
                    SELECT COUNT(*) FROM validation_expectations 
                    WHERE run_id = :run_id
                """)).params(run_id=run.id).scalar()
                
                if expectations == 0:
                    logging.error(f"Run {run.id} has no expectations")
                    return False
                    
                # Verify run has associated metrics
                metrics = self.db.Session().query(text("""
                    SELECT COUNT(*) FROM data_quality_metrics 
                    WHERE metadata->>'validation_run_id' = :run_id
                """)).params(run_id=str(run.id)).scalar()
                
                if metrics == 0:
                    logging.error(f"Run {run.id} has no metrics")
                    return False
                    
            logging.info("Validation data is synchronized")
            return True
            
        except Exception as e:
            logging.error(f"Error checking synchronization: {str(e)}")
            return False
            
    def check_data_quality(self):
        """Check data quality metrics for anomalies."""
        try:
            # Get recent metrics
            recent_metrics = self.db.get_quality_metrics()
            if not recent_metrics:
                logging.warning("No quality metrics found")
                return False
                
            # Group metrics by source and type
            metrics_by_source = {}
            for metric in recent_metrics:
                if metric.source not in metrics_by_source:
                    metrics_by_source[metric.source] = {}
                if metric.metric_name not in metrics_by_source[metric.source]:
                    metrics_by_source[metric.source][metric.metric_name] = []
                metrics_by_source[metric.source][metric.metric_name].append(metric)
                
            # Check for anomalies in each metric
            for source, metrics in metrics_by_source.items():
                for metric_name, values in metrics.items():
                    if len(values) < 2:
                        continue
                        
                    # Calculate moving average
                    recent_values = [m.metric_value for m in values[-5:]]
                    avg = sum(recent_values) / len(recent_values)
                    
                    # Check for significant deviations
                    latest_value = values[-1].metric_value
                    deviation = abs(latest_value - avg) / avg if avg != 0 else 0
                    
                    if deviation > 0.2:  # 20% deviation threshold
                        logging.warning(
                            f"Anomaly detected in {source} {metric_name}: "
                            f"latest={latest_value}, avg={avg:.2f}, deviation={deviation:.2%}"
                        )
                        
            logging.info("Data quality check completed")
            return True
            
        except Exception as e:
            logging.error(f"Error checking data quality: {str(e)}")
            return False
            
    def run_checks(self):
        """Run all synchronization checks."""
        validation_sync = self.check_validation_sync()
        data_quality = self.check_data_quality()
        
        return {
            'validation_sync': validation_sync,
            'data_quality': data_quality,
            'timestamp': datetime.utcnow()
        } 