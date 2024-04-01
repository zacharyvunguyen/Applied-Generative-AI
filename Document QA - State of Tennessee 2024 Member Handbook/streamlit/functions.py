import os
import sys
# Standard library imports for basic operations and concurrency
import os
import io
import json
import base64
import requests
import concurrent.futures
import time
import asyncio
from google.cloud.exceptions import NotFound

# PDF manipulation and IPython display utilities
import PyPDF2
import IPython
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw

# Imaging libraries for image manipulation
from PIL import Image, ImageFont, ImageDraw
import shapely

# Data manipulation and scientific computing libraries
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Vertex AI for machine learning models and Google Cloud services for storage and processing
import vertexai.language_models  # PaLM and Codey Models
import vertexai.generative_models  # for Gemini Models
from google.cloud import documentai, storage, bigquery
from google.api_core import retry
from google.cloud.exceptions import NotFound
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

def set_gcp_credentials_from_file(credential_path):
    if not os.path.exists(credential_path):
        raise FileNotFoundError(f"Service account key file not found at: {credential_path}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
    print("GCP credentials have been set successfully.")


def create_gcs_bucket(project_id, bucket_name, region):
    storage_client = storage.Client(project=project_id)

    # Check if the bucket already exists
    try:
        existing_bucket = storage_client.get_bucket(bucket_name)
        print(f"Bucket {existing_bucket.name} already exists.")
        return existing_bucket
    except NotFound:
        # If the bucket does not exist, proceed to create it
        bucket = storage_client.bucket(bucket_name)
        new_bucket = storage_client.create_bucket(bucket, location=region)
        print(f"Bucket {new_bucket.name} created.")
        return new_bucket


def create_bq_dataset_and_table(project_id, dataset_id, table_id, region):
    """
    Creates a BigQuery dataset and table within it if they do not exist.

    Parameters:
    - project_id: str. The GCP project ID.
    - dataset_id: str. The dataset ID within the project.
    - table_id: str. The table ID within the dataset.
    - region: str. The location for the dataset.

    Returns:
    - The BigQuery Table object for the newly created or existing table.
    """
    bq_client = bigquery.Client(project=project_id)

    # Dataset reference
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)

    # Attempt to get or create the dataset
    try:
        dataset = bq_client.get_dataset(dataset_ref)
        print(f"Dataset {dataset.dataset_id} already exists.")
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = region
        dataset = bq_client.create_dataset(dataset)
        print(f"Dataset {dataset.dataset_id} created.")

    # Table reference
    table_ref = dataset_ref.table(table_id)

    # Attempt to get or create the table
    try:
        table = bq_client.get_table(table_ref)
        print(f"Table {table.table_id} already exists.")
    except NotFound:
        # Define the table schema
        schema = [
            bigquery.SchemaField("example_field", "STRING", mode="NULLABLE")
        ]
        table = bigquery.Table(table_ref, schema=schema)
        table = bq_client.create_table(table)
        print(f"Table {table.table_id} created.")

    # Construct and print the direct link to the table for easy access
    table_link = f"https://console.cloud.google.com/bigquery?project={project_id}&p={project_id}&d={dataset_id}&t={table_id}&page=table"
    print(f"Access your table directly: {table_link}")

    return table

