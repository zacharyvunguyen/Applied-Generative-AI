import streamlit as st
import os
import sys
import io
import json
import base64
import requests
import time
import asyncio
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shapely
from sklearn.manifold import TSNE
from google.cloud import storage, bigquery, documentai
from google.cloud.exceptions import NotFound
from google.api_core import retry
import vertexai
from vertexai import generative_models, language_models

from functions import (
    set_gcp_credentials_from_file,
    create_gcs_bucket,
    create_bq_dataset_and_table,
)  # Import utility functions

# --- Configuration ---
GOOGLE_APPLICATION_CREDENTIALS_PATH = "/Users/zacharynguyen/Documents/GitHub/2024/Applied-Generative-AI/IAM/zacharynguyen-genai-656c475b142a.json"
PROJECT_ID = 'zacharynguyen-genai'
REGION = 'us-central1'
EXPERIMENT = 'cigna-handbook'
SERIES = 'applied-genai-v3'
GCS_BUCKET = PROJECT_ID
BQ_DATASET = SERIES.replace('-', '_')
BQ_TABLE = EXPERIMENT
BQ_REGION = REGION[:2]

# --- Initialization ---
try:
    set_gcp_credentials_from_file(GOOGLE_APPLICATION_CREDENTIALS_PATH)
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

vertexai.init(project=PROJECT_ID, location=REGION)

# --- Streamlit UI ---
model_choice = st.selectbox("Choose a generative model:", ["PaLM", "Gemini"])
question = st.text_area("Enter your query:", "How are emergency services covered, especially if the provider is out-of-network?")

if st.button("Analyze"):
    # --- Model logic ---
    if model_choice == "PaLM":
        model = language_models.TextGenerationModel.from_pretrained('text-bison')
        response = model.predict(question)
        st.write("PaLM Model Response:")
        st.write(response.text)

    elif model_choice == "Gemini":
        model = generative_models.GenerativeModel("gemini-1.0-pro")
        response = model.generate_content(question)
        st.write("Gemini Model Response:")
        st.write(response.text)
