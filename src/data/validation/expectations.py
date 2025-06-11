"""
Data validation module using Great Expectations.
"""
import os
from pathlib import Path
from typing import Dict, List, Any
import great_expectations as ge
from great_expectations.core import ExpectationSuite
from great_expectations.data_context import DataContext
from great_expectations.dataset import PandasDataset
import pandas as pd
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DataValidator:
    def __init__(self, context_root_dir: str = "data/validation"):
        """
        Initialize the data validator.
        
        Args:
            context_root_dir: Directory for Great Expectations context
        """
        self.context_root_dir = Path(context_root_dir)
        self.context_root_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Great Expectations context
        self.context = DataContext.create(
            project_root_dir=str(self.context_root_dir),
            data_docs_sites={
                "local_site": {
                    "class_name": "SiteBuilder",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": str(self.context_root_dir / "data_docs")
                    }
                }
            }
        )
        
        # Create expectation suites for different data sources
        self.suites = {
            'gutenberg': self._create_gutenberg_suite(),
            'wikipedia': self._create_wikipedia_suite(),
            'arxiv': self._create_arxiv_suite()
        }
        
    def _create_gutenberg_suite(self) -> ExpectationSuite:
        """Create expectation suite for Gutenberg data."""
        suite = self.context.create_expectation_suite(
            expectation_suite_name="gutenberg_suite",
            overwrite_existing=True
        )
        
        # Add expectations
        suite.add_expectation_config({
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": "text"}
        })
        suite.add_expectation_config({
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "text"}
        })
        suite.add_expectation_config({
            "expectation_type": "expect_column_value_lengths_to_be_between",
            "kwargs": {
                "column": "text",
                "min_value": int(os.getenv('MIN_TEXT_LENGTH', '100')),
                "max_value": int(os.getenv('MAX_TEXT_LENGTH', '10000'))
            }
        })
        
        return suite
        
    def _create_wikipedia_suite(self) -> ExpectationSuite:
        """Create expectation suite for Wikipedia data."""
        suite = self.context.create_expectation_suite(
            expectation_suite_name="wikipedia_suite",
            overwrite_existing=True
        )
        
        # Add expectations
        suite.add_expectation_config({
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": "text"}
        })
        suite.add_expectation_config({
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "text"}
        })
        suite.add_expectation_config({
            "expectation_type": "expect_column_value_lengths_to_be_between",
            "kwargs": {
                "column": "text",
                "min_value": int(os.getenv('MIN_TEXT_LENGTH', '100')),
                "max_value": int(os.getenv('MAX_TEXT_LENGTH', '10000'))
            }
        })
        
        return suite
        
    def _create_arxiv_suite(self) -> ExpectationSuite:
        """Create expectation suite for arXiv data."""
        suite = self.context.create_expectation_suite(
            expectation_suite_name="arxiv_suite",
            overwrite_existing=True
        )
        
        # Add expectations
        suite.add_expectation_config({
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": "text"}
        })
        suite.add_expectation_config({
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "text"}
        })
        suite.add_expectation_config({
            "expectation_type": "expect_column_value_lengths_to_be_between",
            "kwargs": {
                "column": "text",
                "min_value": int(os.getenv('MIN_TEXT_LENGTH', '100')),
                "max_value": int(os.getenv('MAX_TEXT_LENGTH', '10000'))
            }
        })
        
        return suite
        
    def validate_data(self, data: pd.DataFrame, source: str) -> Dict[str, Any]:
        """
        Validate data against expectations.
        
        Args:
            data: DataFrame to validate
            source: Data source name (gutenberg, wikipedia, or arxiv)
            
        Returns:
            Dictionary containing validation results
        """
        if source not in self.suites:
            raise ValueError(f"Unknown data source: {source}")
            
        # Create dataset
        dataset = PandasDataset(data)
        
        # Run validation
        validation_result = self.context.run_validation_operator(
            "action_list_operator",
            assets_to_validate=[(dataset, self.suites[source])]
        )
        
        # Generate documentation
        self.context.build_data_docs()
        
        return {
            "success": validation_result.success,
            "statistics": validation_result.statistics,
            "results": validation_result.results,
            "data_docs_url": str(self.context_root_dir / "data_docs" / "local_site" / "index.html")
        }
        
    def get_validation_report(self, source: str) -> Dict[str, Any]:
        """
        Get validation report for a data source.
        
        Args:
            source: Data source name
            
        Returns:
            Dictionary containing validation report
        """
        if source not in self.suites:
            raise ValueError(f"Unknown data source: {source}")
            
        return {
            "suite_name": self.suites[source].expectation_suite_name,
            "expectations": [
                {
                    "expectation_type": exp.expectation_type,
                    "kwargs": exp.kwargs
                }
                for exp in self.suites[source].expectations
            ]
        } 