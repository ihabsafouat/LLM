"""
Script to generate data validation reports.
"""
import os
import logging
from pathlib import Path
import json
from datetime import datetime
from dotenv import load_dotenv
from src.data.validation.expectations import DataValidator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'validation_report.log')),
        logging.StreamHandler()
    ]
)

def generate_report():
    """Generate validation report for all data sources."""
    # Initialize validator
    validator = DataValidator()
    
    # Get validation reports for each source
    reports = {
        'gutenberg': validator.get_validation_report('gutenberg'),
        'wikipedia': validator.get_validation_report('wikipedia'),
        'arxiv': validator.get_validation_report('arxiv')
    }
    
    # Add metadata
    report = {
        'timestamp': datetime.now().isoformat(),
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'reports': reports
    }
    
    # Save report
    report_dir = Path('data/validation/reports')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    logging.info(f"Validation report generated: {report_file}")
    
    # Generate HTML report
    validator.context.build_data_docs()
    logging.info(f"Data docs available at: {validator.context_root_dir}/data_docs/local_site/index.html")

if __name__ == "__main__":
    generate_report() 