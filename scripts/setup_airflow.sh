#!/bin/bash

# Create necessary directories
mkdir -p dags logs plugins config

# Copy DAG file
cp src/data/pipeline/dags/data_pipeline_dag.py dags/

# Create Airflow user
docker-compose run airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Initialize the database
docker-compose run airflow-webserver airflow db init

# Start the services
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Initialize connections
chmod +x scripts/init_connections.sh
./scripts/init_connections.sh

echo "Airflow is starting up..."
echo "Web UI will be available at http://localhost:8080"
echo "MinIO Console will be available at http://localhost:9001"
echo "Spark UI will be available at http://localhost:8081" 