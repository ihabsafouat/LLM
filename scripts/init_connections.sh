#!/bin/bash

# Wait for Airflow webserver to be ready
echo "Waiting for Airflow webserver..."
while ! curl -s http://localhost:8080/health > /dev/null; do
    sleep 5
done

# Create Spark connection
docker-compose run airflow-webserver airflow connections add 'spark_default' \
    --conn-type 'spark' \
    --conn-host 'spark://spark-master:7077' \
    --conn-extra '{"queue": "default", "deploy-mode": "cluster", "spark-home": "/opt/spark", "spark-binary": "spark-submit"}'

# Create MinIO connection
docker-compose run airflow-webserver airflow connections add 'minio_default' \
    --conn-type 'http' \
    --conn-host 'http://minio:9000' \
    --conn-login 'minioadmin' \
    --conn-password 'minioadmin'

echo "Airflow connections initialized successfully!" 