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

from functions import set_gcp_credentials_from_file, create_gcs_bucket, create_bq_dataset_and_table, bq_table_check
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
#response = text_model_b.predict(question)
#print("PALM RESPONSE:")
#print(response.text)
#
#response = gemini_text.generate_content(question)
#print("GEMINI RESPONSE")
#print(response.text)

bq_table_check(f'{BQ_DATASET}.{BQ_TABLE}_files_pages'), bq_table_check(f'{BQ_DATASET}.{BQ_TABLE}_files_pages_chunks')

if USE_PRIOR_RUN == False:
    PRIOR_PARSE = False

    # do a check for prior run and present message if found letting user know the prior result exists but not being used
    if RETRIEVE_FROM == 'GCS' and len(list(bucket.list_blobs(prefix=f'{SERIES}/{EXPERIMENT}/files_pages.json'))) > 0:
        print(
            f'Previous results exists in GCS but forcing the creation of new parsing with USE_PRIOR_RUN = {USE_PRIOR_RUN}')
    elif RETRIEVE_FROM == 'BQ' and bq_table_check(f'{BQ_DATASET}.{BQ_TABLE}_files_pages'):
        print(
            f'Previous results exists in BQ but forcing the creation of new parsing with USE_PRIOR_RUN = {USE_PRIOR_RUN}')

elif RETRIEVE_FROM == 'GCS' and len(list(bucket.list_blobs(prefix=f'{SERIES}/{EXPERIMENT}/files_pages.json'))) > 0:
    print(f'Detected {SERIES}/{EXPERIMENT}/files_pages.json')
    print('Importing previous run from GCS')

    # load files_pages: the file+page level information including docai responses in `parsing`
    blob = bucket.blob(f'{SERIES}/{EXPERIMENT}/files_pages.json')
    files_pages = [json.loads(line) for line in blob.download_as_text().splitlines()]
    print(f'Loaded {SERIES}/{EXPERIMENT}/files_pages.json')
    # load files_pages_chunks: the chunks parsed from the files+pages
    blob = bucket.blob(f'{SERIES}/{EXPERIMENT}/files_pages_chunks.json')
    files_pages_chunks = [json.loads(line) for line in blob.download_as_text().splitlines()]
    print(f'Loaded {SERIES}/{EXPERIMENT}/files_pages_chunks.json')
    # Set Indicator to prevent redoing the parsing later in this notebook
    PRIOR_PARSE = True
    print(f'PRIOR_PARSE: {PRIOR_PARSE}')

elif RETRIEVE_FROM == 'BQ' and bq_table_check(f'{BQ_DATASET}.{BQ_TABLE}_files_pages'):
    print('Importing previous run from BigQuery')

    # load files_pages: the file+page level information including docai responses in `parsing`
    files_pages = bq.query(
        f'SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}_files_pages` ORDER BY file_index, page_index').to_dataframe().to_dict(
        'records')
    # convert json string to dictionary:
    for page in files_pages:
        page['parsing'] = json.loads(page['parsing'])

    # load files_pages_chunks: the chunks parsed from the files+pages
    files_pages_chunks = bq.query(
        f'SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}_files_pages_chunks`').to_dataframe().to_dict('records')
    # convert json string to dictionary:
    for chunk in files_pages_chunks:
        chunk['metadata'] = json.loads(chunk['metadata'])
    # sort chunk by file, page, chunk number:
    files_pages_chunks = sorted(files_pages_chunks, key=lambda x: (
    x['metadata']['file_index'], x['metadata']['page_index'], x['metadata']['chunk']))

    # Set Indicator to prevent redoing the parsing later in this notebook
    PRIOR_PARSE = True

else:
    print('No previous run available to import')
    PRIOR_PARSE = False

PARSER_DISPLAY_NAME = 'my_general_processor'
PARSER_TYPE = 'FORM_PARSER_PROCESSOR'
PARSER_VERSION = 'pretrained-form-parser-v2.1-2023-06-26'

for p in docai_client.list_processors(parent = f'projects/{PROJECT_ID}/locations/{LOCATION}'):
    if p.display_name == PARSER_DISPLAY_NAME:
        parser = p
try:
    print('Retrieved existing parser: ', parser.name)
except Exception:
    parser = docai_client.create_processor(
        parent = f'projects/{PROJECT_ID}/locations/{LOCATION}',
        processor = dict(display_name = PARSER_DISPLAY_NAME, type_ = PARSER_TYPE, default_processor_version = PARSER_VERSION)
    )
    print('Created New Parser: ', parser.name)

print("#####Processing the document:#####")
if PRIOR_PARSE:
    print('Using Prior Results')
else:
    document_locations = []
    for source_document in source_documents:
        if source_document.startswith('http'):
            document_locations.append('URL')
            print(f'Use requests to get online document: {source_document}')
        elif source_document.startswith('gs'):
            document_locations.append('GCS')
            print(f'Use GCS to get document in GCS: {source_document}')
        else:
            document_locations.append('UNKNOWN')
            print(f'The source_document variable points to a document in an unknown location type (not gs:// or http): {source_document}')

if PRIOR_PARSE:
    print('Using prior results; no need to import documents again.')
else:
    imported_documents = []
    for s, source_document in enumerate(source_documents):
        location_type = document_locations[s]
        try:
            if location_type == 'URL':
                document_content = requests.get(source_document).content
                print(f'Successfully imported document from URL: {source_document}')
            elif location_type == 'GCS':
                blob_path = source_document.split(f'gs://{GCS_BUCKET}/')[1]
                blob = bucket.blob(blob_path)
                document_content = blob.download_as_bytes()
                print(f'Successfully downloaded document from GCS: {source_document}')
            elif location_type == 'UNKNOWN':
                document_content = None
                print(f'Could not import document, unknown source location: {source_document}')
            else:
                raise ValueError(f"Unhandled document location type: {location_type}")

            imported_documents.append(document_content)
        except Exception as e:
            print(f"Error processing document '{source_document}': {e}")
            imported_documents.append(None)

    if imported_documents:
        print(f'Type of the first imported document: {type(imported_documents[0])}')
    else:
        print('No documents were imported.')

if PRIOR_PARSE:
    print('Using prior results. No need for document conversion.')
else:
    converted_documents = []
    for index, imported_document in enumerate(imported_documents):
        if imported_document:
            try:
                document_reader = PyPDF2.PdfReader(io.BytesIO(imported_document))
                converted_documents.append(document_reader)
                print(f'Document {index + 1} successfully converted for processing.')
            except Exception as e:
                print(f'Error converting document {index + 1}: {e}')
                converted_documents.append(None)
        else:
            print(f'Document {index + 1} is unavailable and cannot be converted.')
            converted_documents.append(None)

    # Ensure there's at least one document to check the type of
    if converted_documents:
        print(f'Type of the first converted document: {type(converted_documents[0])}')
    else:
        print('No documents were converted.')

if PRIOR_PARSE:
    print('Using prior results. No need to analyze document pages.')
else:
    # Ensure there are converted documents to process
    if not converted_documents:
        print("No converted documents available for page analysis.")
    else:
        for index, file in enumerate(converted_documents):
            if file:
                try:
                    num_pages = len(file.pages)
                    print(f"Document {index + 1} ({source_documents[index]}) has {num_pages} pages.")
                except Exception as e:
                    print(f"Error accessing pages in document {index + 1} ({source_documents[index]}): {e}")
            else:
                print(f"Document {index + 1} ({source_documents[index]}) could not be converted or is unavailable.")

if PRIOR_PARSE:
    print('Using Prior Results')
else:
    # Initialize an empty list to store dictionaries containing file index, page number, and page content
    files_pages = []

    # Check if there are any converted documents to process
    if not converted_documents:
        print("No documents available for page extraction.")
    else:
        for file_index, converted_document in enumerate(converted_documents):
            # Verify the converted document is not None
            if converted_document:
                for page_index, page in enumerate(converted_document.pages, start=1):
                    writer = PyPDF2.PdfWriter()
                    writer.add_page(page)

                    # Using io.BytesIO() to capture the page content as bytes
                    with io.BytesIO() as bytes_stream:
                        writer.write(bytes_stream)
                        bytes_stream.seek(0)
                        # Append a dictionary with the extracted data for each page
                        files_pages.append({
                            'file_index': file_index,
                            'page_index': page_index,
                            'raw_file_page': bytes_stream.read()
                        })
                print(f"Processed {len(converted_document.pages)} pages from document {file_index + 1}.")
            else:
                print(f"Document {file_index + 1} is unavailable or could not be converted.")

    # Print the total number of pages processed across all documents
    print(f"Total pages processed: {len(files_pages)}.")


async def docai_runner(files_pages, limit_concur_requests=120):
    limit = asyncio.Semaphore(limit_concur_requests)
    results = [None] * len(files_pages)

    # make requests - async
    async def make_request(p):

        async with limit:
            if limit.locked():
                await asyncio.sleep(0.01)

            ########### manual Error Handling ############################################
            fail_count = 0
            while fail_count <= 20:
                try:
                    result = await docai_async_client.process_document(
                        request=dict(
                            raw_document=documentai.RawDocument(
                                content=files_pages[p]['raw_file_page'],
                                mime_type='application/pdf'
                            ),
                            name=parser.name
                        )
                    )
                    if fail_count > 0:
                        print(f'Item {p} succeeded after fail count = {fail_count}')
                    break
                except:
                    fail_count += 1
                    # print(f'Item {p} failed: current fail count = {fail_count}')
                    await asyncio.sleep(2 ^ (min(fail_count, 6) - 1))
            ##############################################################################

        results[p] = documentai.Document.to_dict(result.document)

    # manage tasks
    tasks = [asyncio.create_task(make_request(p)) for p in range(len(files_pages))]
    responses = await asyncio.gather(*tasks)

    # add parsing to input list of dictionaries for all the pages
    for c, content in enumerate(files_pages):
        content['parsing'] = results[c]

    return

