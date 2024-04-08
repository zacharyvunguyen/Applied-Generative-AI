import os
# Import Google Cloud and Vertex AI libraries
from google.cloud import bigquery
import vertexai
import vertexai.language_models
import vertexai.preview.generative_models
from functions import initial_query,codechat_start,fix_query,answer_question,BQ_QA

# Set environment variables for Google Cloud credentials and TensorFlow logging level
GOOGLE_APPLICATION_CREDENTIALS = "/Users/zacharynguyen/Documents/GitHub/2024/Applied-Generative-AI/IAM/zacharynguyen-genai-656c475b142a.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define Google Cloud project and region information
PROJECT_ID = 'zacharynguyen-genai'
REGION = 'us-central1'
EXPERIMENT = 'sdoh_cdc_wonder_natality'
SERIES = 'applied-genai'

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# Initialize the BigQuery client
bq = bigquery.Client(project=PROJECT_ID)

# Initialize Vertex AI models
textgen_model = vertexai.language_models.TextGenerationModel.from_pretrained('text-bison@002')
codegen_model = vertexai.language_models.CodeGenerationModel.from_pretrained('code-bison@002')
gemini_model = vertexai.preview.generative_models.GenerativeModel("gemini-pro")

# BigQuery constants for querying
BQ_PROJECT = 'bigquery-public-data'
BQ_DATASET = 'sdoh_cdc_wonder_natality'
BQ_TABLES = ['county_natality', 'county_natality_by_mother_race', 'county_natality_by_father_race']

# Construct and execute a BigQuery query to retrieve schema columns
query = f"""
    SELECT * EXCEPT(field_path, collation_name, rounding_mode)
    FROM `{BQ_PROJECT}.{BQ_DATASET}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
    WHERE table_name in ({','.join([f'"{table}"' for table in BQ_TABLES])})
"""
schema_columns = bq.query(query=query).to_dataframe()
# Define the question for the model
question = "Which mother's single race category reports the highest average number of births and what is that number?"
section = BQ_QA(question)
