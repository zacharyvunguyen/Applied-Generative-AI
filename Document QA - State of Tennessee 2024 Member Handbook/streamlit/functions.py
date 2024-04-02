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


# Project and Data Analysis Settings
PROJECT_ID = 'zacharynguyen-genai'
REGION = 'us-central1'
EXPERIMENT = 'cigna-handbook'
SERIES = 'applied-genai-v3'

# Data Storage and Retrieval Configuration
SAVE_IN = 'ALL'  # Options: GCS, BQ, ALL
RETRIEVE_FROM = 'GCS'  # Options: GCS, BQ. Default action is to parse and embed if not present.

# Google Cloud Storage (GCS) Setup
GCS_BUCKET = PROJECT_ID  # Naming the bucket after the project ID for consistency

# BigQuery (BQ) Setup for Storing Results
BQ_PROJECT = PROJECT_ID
BQ_DATASET = SERIES.replace('-', '_')  # Formatting to comply with BQ naming conventions
BQ_TABLE = EXPERIMENT
BQ_REGION = REGION[:2]  # Simplified regional code derived from the full region string

# Document Source Configuration
# Specify the locations of source documents to be processed
source_documents = [
    'https://www.tn.gov/content/dam/tn/partnersforhealth/documents/cigna_member_handbook_2024.pdf'
]


# Initialize BigQuery client
bq = bigquery.Client(project=PROJECT_ID)

# Initialize Google Cloud Storage (GCS) client and get the bucket
gcs = storage.Client(project=PROJECT_ID)
bucket = gcs.bucket(GCS_BUCKET)



def set_gcp_credentials_from_file(credential_path):
    if not os.path.exists(credential_path):
        raise FileNotFoundError(f"Service account key file not found at: {credential_path}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
    print("GCP credentials have been set successfully.")

GOOGLE_APPLICATION_CREDENTIALS_PATH = "/Users/zacharynguyen/Documents/GitHub/2024/Applied-Generative-AI/IAM/zacharynguyen-genai-656c475b142a.json"
try:
    set_gcp_credentials_from_file(GOOGLE_APPLICATION_CREDENTIALS_PATH)
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

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


def bq_table_check(table):
    from google.cloud.exceptions import NotFound
    try:
        bq.get_table(table)
        print(f'Table "{table}" found')
        return True
    except NotFound:
        print(f'Table "{table}" not found')
        return False


