"""
Script to clean up temporary files and directories.
"""
import os
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'cleanup.log')),
        logging.StreamHandler()
    ]
)

def cleanup():
    """Clean up temporary files and directories."""
    # Directories to clean
    temp_dirs = [
        Path('data/temp'),
        Path('data/raw'),
        Path('data/processed'),
        Path('logs'),
    ]
    
    # Clean each directory
    for dir_path in temp_dirs:
        if dir_path.exists():
            logging.info(f"Cleaning directory: {dir_path}")
            shutil.rmtree(dir_path)
            dir_path.mkdir(exist_ok=True)
            
    # Clean log files
    log_files = Path('.').glob('*.log')
    for log_file in log_files:
        if log_file.exists():
            logging.info(f"Cleaning log file: {log_file}")
            log_file.unlink()
            
    logging.info("Cleanup completed")

if __name__ == "__main__":
    cleanup() 