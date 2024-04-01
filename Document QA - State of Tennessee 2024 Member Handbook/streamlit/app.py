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

from functions import set_gcp_credentials_from_file, create_gcs_bucket, create_bq_dataset_and_table
# Path to your service account key file
GOOGLE_APPLICATION_CREDENTIALS_PATH = "/Users/zacharynguyen/Documents/GitHub/2024/Applied-Generative-AI/IAM/zacharynguyen-genai-656c475b142a.json"
try:
    set_gcp_credentials_from_file(GOOGLE_APPLICATION_CREDENTIALS_PATH)
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

# Configuration for Project Environment and Data Handling

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

# Prior Run Handling
# Determines whether to use data from a previous run based on the USE_PRIOR_RUN flag
USE_PRIOR_RUN = True  # Boolean flag to indicate preference for reusing previous data when available

# Initial Analysis Query
# Defining the first question to guide the analysis or processing of the document
question = "How are emergency services covered, especially if the provider is out-of-network?"

print(f"GOOGLE_APPLICATION_CREDENTIALS_PATH: {GOOGLE_APPLICATION_CREDENTIALS_PATH}")
print(f"PROJECT_ID: {PROJECT_ID}")
print(f"REGION: {REGION}")
print(f"EXPERIMENT: {EXPERIMENT}")
print(f"SERIES: {SERIES}")
print(f"SAVE_IN: {SAVE_IN}")
print(f"RETRIEVE_FROM: {RETRIEVE_FROM}")
print(f"GCS_BUCKET: {GCS_BUCKET}")
print(f"BQ_PROJECT: {BQ_PROJECT}")
print(f"BQ_DATASET: {BQ_DATASET}")
print(f"BQ_TABLE: {BQ_TABLE}")
print(f"BQ_REGION: {BQ_REGION}")
print(f"source_documents: {source_documents}")
print(f"USE_PRIOR_RUN: {USE_PRIOR_RUN}")
print(f"question: {question}")

# Create GCS Bucket
create_gcs_bucket(PROJECT_ID, GCS_BUCKET, REGION)

# Create BigQuery Dataset and Table
create_bq_dataset_and_table(PROJECT_ID, BQ_DATASET, BQ_TABLE, BQ_REGION)

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# Setup Document AI clients
LOCATION = REGION.split('-')[0]
docai_endpoint = f"{LOCATION}-documentai.googleapis.com"
docai_client_options = {"api_endpoint": docai_endpoint}

# Document AI synchronous client
docai_client = documentai.DocumentProcessorServiceClient(client_options=docai_client_options)

# Document AI asynchronous client
docai_async_client = documentai.DocumentProcessorServiceAsyncClient(client_options=docai_client_options)

# Initialize BigQuery client
bq = bigquery.Client(project=PROJECT_ID)

# Initialize Google Cloud Storage (GCS) client and get the bucket
gcs = storage.Client(project=PROJECT_ID)
bucket = gcs.bucket(GCS_BUCKET)

# Print confirmation that clients have been initialized successfully
print("Initialized Vertex AI, Document AI, BigQuery, and GCS clients successfully.")

# Import Vertex AI models for generative tasks and language processing
from vertexai import generative_models, language_models

# Initialize Gemini Model for advanced generative tasks
# Gemini models are designed for a wide range of generative AI applications
gemini_text = generative_models.GenerativeModel("gemini-1.0-pro")

# Initialize PaLM Models for language processing and generation
# Text Embedding Model for converting text into high-dimensional vectors
textembed_model = language_models.TextEmbeddingModel.from_pretrained('textembedding-gecko')

# Text Generation Models for generating coherent and contextually relevant text
# Bison Model - Standard version for text generation
text_model_b = language_models.TextGenerationModel.from_pretrained('text-bison')

# Bison Model - Extended version with support for longer text sequences
text_model_b32 = language_models.TextGenerationModel.from_pretrained('text-bison-32k')

# Unicorn Model - Versatile model for a broad range of text generation tasks
text_model_u = language_models.TextGenerationModel.from_pretrained('text-unicorn')

# Prediction using the standard text generation model (Bison Model)
response = text_model_b.predict(question)
print("PALM RESPONSE:")
print(response.text)

response = gemini_text.generate_content(question)
print("GEMINI RESPONSE")
print(response.text)