"""
Database manager for validation metadata.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database
from dotenv import load_dotenv
from .models import Base, ValidationRun, ValidationExpectation, DataQualityMetric

# Load environment variables
load_dotenv()

class DatabaseManager:
    def __init__(self):
        """Initialize database connection."""
        # Use PostgreSQL connection from Airflow environment
        self.db_url = os.getenv('POSTGRES_URI', 'postgresql://airflow:airflow@postgres/airflow')
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
    def store_validation_run(self, source: str, success: bool, statistics: dict, 
                           data_docs_url: str, environment: str) -> ValidationRun:
        """Store validation run metadata."""
        session = self.Session()
        try:
            run = ValidationRun(
                source=source,
                success=success,
                statistics=statistics,
                data_docs_url=data_docs_url,
                environment=environment
            )
            session.add(run)
            session.commit()
            return run
        finally:
            session.close()
            
    def store_expectation(self, run_id: int, expectation_type: str, success: bool,
                         kwargs: dict, result: dict) -> ValidationExpectation:
        """Store expectation result."""
        session = self.Session()
        try:
            expectation = ValidationExpectation(
                run_id=run_id,
                expectation_type=expectation_type,
                success=success,
                kwargs=kwargs,
                result=result
            )
            session.add(expectation)
            session.commit()
            return expectation
        finally:
            session.close()
            
    def store_quality_metric(self, source: str, metric_name: str, metric_value: float,
                            threshold: float, passed: bool, metadata: dict = None) -> DataQualityMetric:
        """Store data quality metric."""
        session = self.Session()
        try:
            metric = DataQualityMetric(
                source=source,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold,
                passed=passed,
                metadata=metadata
            )
            session.add(metric)
            session.commit()
            return metric
        finally:
            session.close()
            
    def get_latest_validation_runs(self, limit: int = 10) -> list:
        """Get latest validation runs."""
        session = self.Session()
        try:
            return session.query(ValidationRun)\
                .order_by(ValidationRun.timestamp.desc())\
                .limit(limit)\
                .all()
        finally:
            session.close()
            
    def get_validation_stats(self, source: str = None) -> dict:
        """Get validation statistics."""
        session = self.Session()
        try:
            query = session.query(ValidationRun)
            if source:
                query = query.filter(ValidationRun.source == source)
                
            total = query.count()
            successful = query.filter(ValidationRun.success == True).count()
            
            return {
                'total_runs': total,
                'successful_runs': successful,
                'success_rate': (successful / total * 100) if total > 0 else 0
            }
        finally:
            session.close()
            
    def get_quality_metrics(self, source: str = None, metric_name: str = None) -> list:
        """Get data quality metrics."""
        session = self.Session()
        try:
            query = session.query(DataQualityMetric)
            if source:
                query = query.filter(DataQualityMetric.source == source)
            if metric_name:
                query = query.filter(DataQualityMetric.metric_name == metric_name)
                
            return query.order_by(DataQualityMetric.timestamp.desc()).all()
        finally:
            session.close() 