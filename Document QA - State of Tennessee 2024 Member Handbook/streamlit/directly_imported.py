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
# Define the path to your service account key file
service_account_key_path = "/Users/zacharynguyen/Documents/GitHub/2024/Applied-Generative-AI/IAM/zacharynguyen-genai-656c475b142a.json"

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path

# Verify if the variable is set correctly
print("GOOGLE_APPLICATION_CREDENTIALS is set to:", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

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


USE_PRIOR_RUN = True  # Boolean flag to indicate preference for reusing previous data when available

# Initial Analysis Query
# Defining the first question to guide the analysis or processing of the document
question = "How are emergency services covered, especially if the provider is out-of-network?"


# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# Setup Document AI clients
LOCATION = REGION.split('-')[0]
docai_endpoint = f"{LOCATION}-documentai.googleapis.com"
docai_client_options = {"api_endpoint": docai_endpoint}

# Document AI synchronous client
docai_client = documentai.DocumentProcessorServiceClient(client_options=docai_client_options)

# Document AI asynchronous client
#docai_async_client = documentai.DocumentProcessorServiceAsyncClient(client_options=docai_client_options)

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

chunks_embed_db = np.array([chunk['embedding'] for chunk in files_pages_chunks])


print("FUNCTION FOR BOT")


def get_chunks(query, k=-1, simk=-1):
    # k set the number of matches to retrieve, regarless of similarity. k = -1 will trigger calculating k dynamically.
    # simk sets a threshold for similarity: <=0 uses k, (0,1] will get all matches with similarity in range [1-simk, 1]

    query_embed = np.array(textembed_model.get_embeddings([query])[0].values)
    similarity = np.dot(query_embed,
                        chunks_embed_db.T)  # for dot product, higher is better match, since normalized embeddings 1 is best, 0 is worst
    matches = np.argsort(similarity)[::-1].tolist()

    if k <= 0:
        # algorithm to dynamically pick k
        k = 1 + 3 * int(10 * (1 - similarity[matches[0]]))
    if simk <= 0:
        matches = [(match, similarity[match]) for match in matches[0:k]]
    elif simk > 0 and simk <= 1:
        indicies = np.where(similarity >= 1 - simk)[0]
        matches = [(i, similarity[i]) for i in indicies]

    return matches


def expand_retrieval(contexts, DISTANCE):
    additional_contexts = []
    if DISTANCE > 0:

        # for each page look for surrounding chunks, collect chunks
        chunk_indexes = []
        for context in contexts:
            # get matches for the page from contexts
            matches = get_retrieval(context[2], simk=DISTANCE,
                                    file_page=(context[3]['file_index'], context[3]['page_index']))
            for match in matches:
                if match[0] not in chunk_indexes and match[0] not in [c[0] for c in contexts]:
                    chunk_indexes += [match[0]]
                    additional_contexts.append(match)

    return additional_contexts


def get_retrieval(question, k=-1, simk=-1, DISTANCE=0, file_page=None):
    if file_page:  # this is from a call to this function by expand_retrieval
        matches = [match + (files_pages_chunks[match[0]]['text'], files_pages_chunks[match[0]]['metadata'], True) for
                   match in get_chunks(question, k=k, simk=simk) if file_page == (
                   files_pages_chunks[match[0]]['metadata']['file_index'],
                   files_pages_chunks[match[0]]['metadata']['page_index'])]
    else:  # this is from a call to this function by the main function: document_bot
        matches = [match + (files_pages_chunks[match[0]]['text'], files_pages_chunks[match[0]]['metadata'], False) for
                   match in get_chunks(question, k=k, simk=simk)]

    if DISTANCE > 0:
        matches = matches + expand_retrieval(matches, DISTANCE)

    return matches

print("Retrieval Functions-These retrieve context. LOADED")


def get_augmented(question, contexts, ground):
    prompt = ''

    if ground:
        prompt += "Give a detailed answer to the question using only the information from the numbered contexts provided below."
        prompt += "\n\nContexts:\n"
        prompt += "\n".join([f'  * Context {c + 1}: "{context[2]}"' for c, context in enumerate(contexts)])
        prompt += "\n\nQuestion: " + question
    else:
        prompt += "Question: " + question

    # add the trigger to the prompt.  In this case, also include the zero shot chain of thought prompt "think step by step".
    prompt += "\n\nAnswer the question and give and explanation. Think step by step."

    return prompt

print("Augmentation Functions - This function prepares the prompt by also adding retrieved context = augmenting. LOADED")


def generate_gemini(prompt, genconfigs, model):
    response = model.generate_content(
        prompt,
        generation_config=vertexai.generative_models.GenerationConfig(
            **genconfigs

        )
    )

    try:
        text = response.text
    except Exception:
        text = None

    counter = 0
    while not text:
        genconfigs['temperature'] = .5 - counter * .1
        response = model.generate_content(
            prompt,
            generation_config=vertexai.generative_models.GenerationConfig(
                **genconfigs

            )
        )
        try:
            text = response.text
        except Exception:
            text = None
            counter += 1

        if counter == 6:
            text = 'Please check the prompt, it appears the response is getting blocked.'

    return text


def generate_palm(prompt, genconfigs, model):
    response = model.predict(
        prompt,
        **genconfigs
    )

    return response.text


def get_generation(prompt, max_output_tokens, model):
    models = dict(GEMINI=gemini_text, PALM_BISON=text_model_b, PALM_BISION32k=text_model_b32, PALM_UNICORN=text_model_u)

    genconfigs = dict(max_output_tokens=max_output_tokens)

    if model == 'GEMINI':
        response = generate_gemini(prompt, genconfigs, models[model])
    else:
        response = generate_palm(prompt, genconfigs, models[model])

    return response

print("Generation Functions * These functions interact with LLMs to create responses.LOADED")

# get a font to use for annotating the page images:
# get font for annotations: get fonts from fonts.google.com
font_source_url = "https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap"
font_source = requests.get(font_source_url).content.decode("utf-8")
start_url = font_source.find('url(') + 4
end_url = font_source.find(')', start_url)
font_url = font_source[start_url:end_url]
font = PIL.ImageFont.truetype(io.BytesIO(requests.get(font_url).content), 35)


def get_presentation(question, contexts, DISTANCE, response, display_contexts, display_annotations):
    print(f'**The Question:**\n\n{question}\n\n')
    print(f'**The Response:**\n\n{response}\n\n')

    if display_contexts:
        context_pres = '**Sources:**\n\n'
        for index, context in enumerate(contexts, start=1):
            page_info = next(
                (d for d in files_pages if
                 d['file_index'] == context[3]['file_index'] and d['page_index'] == context[3]['page_index']),
                None)
            if page_info:
                pdf_url = page_info['parsing']['path']
                page_number = page_info['parsing']['page']
                full_url = f"{pdf_url}cigna_member_handbook_2024.pdf#page={page_number}"
                similarity = context[1]
                context_pres += f'- Context {index}: [Page {page_number}, similarity: {similarity:.3f}]({full_url})\n'
        print(context_pres)

    if display_annotations:
        print('**Annotated Document Pages**\n')
        pages = sorted(list(set([(context[3]['file_index'], context[3]['page_index']) for context in contexts])),
                       key=lambda x: (x[0], x[1]))
        for page in pages:
            image_data = next(d['parsing']['pages'][0]['image']['content'] for d in files_pages if
                              d['file_index'] == page[0] and d['page_index'] == page[1])
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            draw = ImageDraw.Draw(image)

            for c, context in enumerate(
                    [ctx for ctx in contexts if (ctx[3]['file_index'], ctx[3]['page_index']) == page]):
                vertices = context[3]['vertices']
                color = 'green' if not context[4] else 'blue'
                prefix = 'Source' if not context[4] else 'Expanded Source'
                similarity_score = context[1]
                # Now include the similarity score in the annotation
                annotation_text = f"{prefix} {c + 1} (Similarity: {similarity_score:.3f})"
                draw.polygon([
                    (vertices[0]['x'], vertices[0]['y']),
                    (vertices[1]['x'], vertices[1]['y']),
                    (vertices[2]['x'], vertices[2]['y']),
                    (vertices[3]['x'], vertices[3]['y'])
                ], outline=color, width=5)
                draw.text((vertices[0]['x'] - 50, vertices[0]['y'] - 50), annotation_text, fill=color, font=font)

            # Save the annotated image
            save_path = f"annotated_page_{page[0]}_{page[1]}.png"
            image.save(save_path)
            print(f"Saved annotated image to {save_path}")

def document_bot(question, max_output_tokens=1000, DISTANCE=0, MODEL='GEMINI', display_contexts=False,
                 display_annotations=False, ground=True):
    # this function directly references (without input): font
    # DISTANCE = .1 # float in [0, 1], 0 return no additional context, 1 return all on unique pages
    # MODEL = 'GEMINI' # one of: GEMINI, PALM_BISON, PALM_BISON32K, PALM_UNICORN

    # R: Retrival
    if ground:
        contexts = get_retrieval(question, DISTANCE=DISTANCE)
    else:
        contexts = []

    # A: Augemented
    prompt = get_augmented(question, contexts, ground)

    # G: Generation
    response = get_generation(prompt, max_output_tokens, MODEL)

    # Present Answer
    get_presentation(question, contexts, DISTANCE, response, display_contexts, display_annotations)

    return prompt

question ="when to seek for urgencare instead of emergency ?"
#prompt = document_bot(question, display_contexts = True, display_annotations = True)
prompt = document_bot(question, max_output_tokens=1000, DISTANCE=0.2, MODEL='GEMINI', display_contexts=True,
                 display_annotations=True, ground=False)
# DISTANCE = .1 # float in [0, 1], 0 return no additional context, 1 return all on unique pages
    # MODEL = 'GEMINI' # one of: GEMINI, PALM_BISON, PALM_BISON32K, PALM_UNICORN