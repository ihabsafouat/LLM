import os
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import DataContext
from great_expectations.data_context.types.base import DataContextConfig, FilesystemStoreBackendDefaults

def setup_great_expectations():
    # Create the base directory
    os.makedirs("great_expectations", exist_ok=True)
    
    # Create the configuration
    base_dir = os.path.abspath("great_expectations")
    data_context_config = DataContextConfig(
        config_version=3.0,
        datasources={
            "pandas_datasource": {
                "class_name": "Datasource",
                "module_name": "great_expectations.datasource",
                "execution_engine": {
                    "class_name": "PandasExecutionEngine",
                    "module_name": "great_expectations.execution_engine",
                },
                "data_connectors": {
                    "runtime_data_connector": {
                        "class_name": "RuntimeDataConnector",
                        "module_name": "great_expectations.datasource.data_connector",
                        "batch_identifiers": ["default_identifier_name"],
                    }
                }
            }
        },
        stores={
            "expectations_store": {
                "class_name": "ExpectationsStore",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": os.path.join(base_dir, "expectations")
                }
            },
            "validations_store": {
                "class_name": "ValidationsStore",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": os.path.join(base_dir, "validations")
                }
            },
            "evaluation_parameter_store": {
                "class_name": "EvaluationParameterStore",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": os.path.join(base_dir, "evaluation_parameters")
                }
            },
            "checkpoint_store": {
                "class_name": "CheckpointStore",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": os.path.join(base_dir, "checkpoints")
                }
            }
        },
        data_docs_sites={
            "local_site": {
                "class_name": "SiteBuilder",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": os.path.join(base_dir, "uncommitted/data_docs/local_site")
                },
                "site_index_builder": {
                    "class_name": "DefaultSiteIndexBuilder"
                }
            }
        },
        validation_operators={
            "action_list_operator": {
                "class_name": "ActionListValidationOperator",
                "action_list": [
                    {
                        "name": "store_validation_result",
                        "action": {
                            "class_name": "StoreValidationResultAction"
                        }
                    },
                    {
                        "name": "store_evaluation_params",
                        "action": {
                            "class_name": "StoreEvaluationParametersAction"
                        }
                    },
                    {
                        "name": "update_data_docs",
                        "action": {
                            "class_name": "UpdateDataDocsAction"
                        }
                    }
                ]
            }
        },
        anonymous_usage_statistics={
            "enabled": False
        },
        expectations_store_name="expectations_store",
        validations_store_name="validations_store",
        evaluation_parameter_store_name="evaluation_parameter_store",
    )
    
    # Write the config to great_expectations/great_expectations.yml
    with open("great_expectations/great_expectations.yml", "w") as f:
        data_context_config.to_yaml(outfile=f)

    # Instantiate the context from the directory
    context = ge.data_context.DataContext("great_expectations")
    
    # Create necessary directories
    os.makedirs("great_expectations/expectations", exist_ok=True)
    os.makedirs("great_expectations/validations", exist_ok=True)
    os.makedirs("great_expectations/evaluation_parameters", exist_ok=True)
    os.makedirs("great_expectations/checkpoints", exist_ok=True)
    os.makedirs("great_expectations/uncommitted/data_docs/local_site", exist_ok=True)
    
    print("Great Expectations configuration has been set up successfully!")

if __name__ == "__main__":
    setup_great_expectations() 