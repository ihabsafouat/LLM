"""
Database models for validation metadata.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class ValidationRun(Base):
    """Model for storing validation run metadata."""
    __tablename__ = 'validation_runs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String, nullable=False)
    success = Column(Boolean, nullable=False)
    statistics = Column(JSON)
    data_docs_url = Column(String)
    environment = Column(String)
    
    # Relationships
    expectations = relationship("ValidationExpectation", back_populates="run")
    
class ValidationExpectation(Base):
    """Model for storing individual expectation results."""
    __tablename__ = 'validation_expectations'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('validation_runs.id'))
    expectation_type = Column(String, nullable=False)
    success = Column(Boolean, nullable=False)
    kwargs = Column(JSON)
    result = Column(JSON)
    
    # Relationships
    run = relationship("ValidationRun", back_populates="expectations")
    
class DataQualityMetric(Base):
    """Model for storing data quality metrics."""
    __tablename__ = 'data_quality_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float)
    threshold = Column(Float)
    passed = Column(Boolean)
    meta_data = Column(JSON) 