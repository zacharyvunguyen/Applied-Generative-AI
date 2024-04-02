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

#print(f"GOOGLE_APPLICATION_CREDENTIALS_PATH: {GOOGLE_APPLICATION_CREDENTIALS_PATH}")
#print(f"PROJECT_ID: {PROJECT_ID}")
#print(f"REGION: {REGION}")
#print(f"EXPERIMENT: {EXPERIMENT}")
#print(f"SERIES: {SERIES}")
#print(f"SAVE_IN: {SAVE_IN}")
#print(f"RETRIEVE_FROM: {RETRIEVE_FROM}")
#print(f"GCS_BUCKET: {GCS_BUCKET}")
#print(f"BQ_PROJECT: {BQ_PROJECT}")
#print(f"BQ_DATASET: {BQ_DATASET}")
#print(f"BQ_TABLE: {BQ_TABLE}")
#print(f"BQ_REGION: {BQ_REGION}")
#print(f"source_documents: {source_documents}")
#print(f"USE_PRIOR_RUN: {USE_PRIOR_RUN}")
#print(f"question: {question}")

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

bq_table_check(f'{BQ_DATASET}.{BQ_TABLE}_files_pages'), bq_table_check(f'{BQ_DATASET}.{BQ_TABLE}_files_pages_chunks')

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


#def get_presentation(question, contexts, DISTANCE, response, display_contexts, display_annotations):
#    # repeat the question
#    print(f'**The Question:**\n\n{question}\n\n')
#    #IPython.display.display(IPython.display.Markdown(f'**The Question:**\n\n{question}\n\n'))

#    # show the answer
#    #IPython.display.display(IPython.display.Markdown(f'**The Response:**\n\n{response}\n\n'))
#    print(f'**The Response:**\n\n{response}\n\n')
#    if display_contexts:
#        # display the contexts information: page, similarity, hyperlink
#        context_pres = '**Sources:**\n\n'
#        pages = []
#        context_types = [c[4] for c in contexts]
#        if DISTANCE > 0:
#            context_pres += f'Note: The {len(contexts) - sum(context_types)} contexts were expanded to gather {sum(context_types)} additional chunks on pages with matches using a similarity distance of {DISTANCE}.\n'
#        for context in contexts:
#            page = next(
#                [d['parsing']['path'], d['parsing']['file'], d['parsing']['page'], d['file_index'], d['page_index']] for
#                d in files_pages if
#                d['file_index'] == context[3]['file_index'] and d['page_index'] == context[3]['page_index'])
#            pages.append(page)
#            if not context[4]:
#                context_pres += f'1. {page[0]}{page[1]}#page={page[2]}\n\t* page: {page[2]}, similarity to question is {context[1]:.3f}\n'
#            # the following is commented out, if uncommented it would also add the expanded contexts to printed list (this can be very long for DISTANCE = 1 which is the full page)
#            # else:
#            #    context_pres += f'1. {page[0]}{page[1]}#page={page[2]}\n\t* page: {page[2]}, similarity to primary context is {context[1]:.3f}\n'
#        #IPython.display.display(IPython.display.Markdown(context_pres))
#        print(f'Context: {context_pres}')

#    if display_annotations:
#        # display each page with annotations
#        IPython.display.display(IPython.display.Markdown('**Annotated Document Pages**\n\n'))
#        # list of unique pages across contexts: sorted list of tuple(file_index, page_index)
#        pages = sorted(list(set([(page[3], page[4]) for page in pages])), key=lambda x: (x[0], x[1]))
#        # list of PIL images for each unique page
#        images = []
#        for page in pages:
#            image = next(d['parsing']['pages'][0]['image']['content'] for d in files_pages if
#                         d['file_index'] == page[0] and d['page_index'] == page[1])
#            images.append(
#                PIL.Image.open(
#                    io.BytesIO(
#                        base64.decodebytes(
#                            image.encode('utf-8')
#                        )
#                    )
#                )
#            )
#        # annotate the contexts on the pages:
#        for c, context in enumerate(contexts):
#            image = images[pages.index((context[3]['file_index'], context[3]['page_index']))]
#            vertices = context[3]['vertices']
#            draw = PIL.ImageDraw.Draw(image)
#            if not context[4]:
#                color = 'green'
#                prefix = 'Source'
#            else:
#                color = 'blue'
#                prefix = 'Expanded Source'
#            draw.polygon([
#                vertices[0]['x'], vertices[0]['y'],
#                vertices[1]['x'], vertices[1]['y'],
#                vertices[2]['x'], vertices[2]['y'],
#                vertices[3]['x'], vertices[3]['y']
#            ], outline=color, width=5)
#            draw.text(
#                xy=(vertices[1]['x'], vertices[1]['y']), text=f"{prefix} {c + 1}", fill=color, anchor='rd', font=font
#            )

#        for image in images:
#            IPython.display.display(image.resize(tuple([int(.25 * x) for x in image.size])))

#    return
def get_presentation(question, contexts, DISTANCE, response, display_contexts, display_annotations):
    print(f'**The Question:**\n\n{question}\n\n')
    print(f'**The Response:**\n\n{response}\n\n')

    if display_contexts:
        context_pres = '**Sources:**\n\n'
        pages = []
        for context in contexts:
            page_info = next(
                (d for d in files_pages if d['file_index'] == context[3]['file_index'] and d['page_index'] == context[3]['page_index']),
                None)
            if page_info:
                page = page_info['parsing']['page']
                context_pres += f'- Page: {page}, similarity: {context[1]:.3f}\n'
                pages.append(page_info)
        print(context_pres)

    if display_annotations:
        print('**Annotated Document Pages**\n')
        for page_info in pages:
            # Assuming the image is encoded in base64 within the 'page_info'
            image_data = base64.b64decode(page_info['parsing']['pages'][0]['image']['content'])
            image = Image.open(io.BytesIO(image_data))
            draw = ImageDraw.Draw(image)

            # Loop through contexts to find those relevant to this page and annotate
            for context in [c for c in contexts if c[3]['file_index'] == page_info['file_index'] and c[3]['page_index'] == page_info['page_index']]:
                vertices = context[3]['vertices']
                color = 'green' if not context[4] else 'blue'
                draw.polygon([
                    (vertices[0]['x'], vertices[0]['y']),
                    (vertices[1]['x'], vertices[1]['y']),
                    (vertices[2]['x'], vertices[2]['y']),
                    (vertices[3]['x'], vertices[3]['y']),
                ], outline=color, width=5)
                # Optional: Add text if needed
                # Ensure you have defined 'font' somewhere, or remove this part
                # draw.text((vertices[0]['x'], vertices[0]['y']), f"{'Source' if not context[4] else 'Expanded Source'}", fill=color, font=font)

            # Save or display the image as needed
            save_path = f"img/annotated_page_{page_info['file_index']}_{page_info['page_index']}.png"
            image.save(save_path)
            print(f"Saved annotated image to {save_path}")
print("Presentation Functions * These prepare the response for presentation - and display the results.LOADED")


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

question ="What is the process for filing an appeal if a claim is denied?"
#prompt = document_bot(question, display_contexts = True, display_annotations = True)
prompt = document_bot(question, max_output_tokens=1000, DISTANCE=0.2, MODEL='GEMINI', display_contexts=True,
                 display_annotations=True, ground=True)
# DISTANCE = .1 # float in [0, 1], 0 return no additional context, 1 return all on unique pages
    # MODEL = 'GEMINI' # one of: GEMINI, PALM_BISON, PALM_BISON32K, PALM_UNICORN