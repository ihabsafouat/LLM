"""
Airflow DAG for data pipeline.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.models import Variable

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='Data pipeline for text data collection and processing',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data', 'pipeline'],
)

# Extract tasks
extract_gutenberg = PythonOperator(
    task_id='extract_gutenberg',
    python_callable='src.data.pipeline.extract_gutenberg:extract_books',
    dag=dag,
)

extract_wikipedia = PythonOperator(
    task_id='extract_wikipedia',
    python_callable='src.data.pipeline.extract_wikipedia:extract_articles',
    dag=dag,
)

extract_arxiv = PythonOperator(
    task_id='extract_arxiv',
    python_callable='src.data.pipeline.extract_arxiv:extract_papers',
    dag=dag,
)

# Transform task using Spark
transform_data = SparkSubmitOperator(
    task_id='transform_data',
    application='src/data/pipeline/transform.py',
    conn_id='spark_default',
    verbose=True,
    conf={
        'spark.executor.memory': '1g',
        'spark.executor.cores': '1',
        'spark.driver.memory': '1g',
        'spark.driver.cores': '1',
        'spark.executor.instances': '1',
        'spark.dynamicAllocation.enabled': 'false'
    },
    dag=dag,
)

# Validation tasks
validate_gutenberg = PythonOperator(
    task_id='validate_gutenberg',
    python_callable='scripts.validate_data:validate_data',
    op_kwargs={'source': 'gutenberg'},
    dag=dag,
)

validate_wikipedia = PythonOperator(
    task_id='validate_wikipedia',
    python_callable='scripts.validate_data:validate_data',
    op_kwargs={'source': 'wikipedia'},
    dag=dag,
)

validate_arxiv = PythonOperator(
    task_id='validate_arxiv',
    python_callable='scripts.validate_data:validate_data',
    op_kwargs={'source': 'arxiv'},
    dag=dag,
)

# Generate validation report
generate_report = PythonOperator(
    task_id='generate_report',
    python_callable='scripts.generate_validation_report:generate_report',
    dag=dag,
)

# Cleanup task
cleanup = PythonOperator(
    task_id='cleanup',
    python_callable='scripts.cleanup:cleanup_data',
    dag=dag,
)

# Set task dependencies
[extract_gutenberg, extract_wikipedia, extract_arxiv] >> transform_data
transform_data >> [validate_gutenberg, validate_wikipedia, validate_arxiv]
[validate_gutenberg, validate_wikipedia, validate_arxiv] >> generate_report
generate_report >> cleanup 