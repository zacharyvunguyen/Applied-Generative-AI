import streamlit as st
import os
# Import Google Cloud and Vertex AI libraries
from google.cloud import bigquery
import vertexai
import vertexai.language_models
import vertexai.preview.generative_models
from functions import initial_query,codechat_start,fix_query,answer_question

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

# Display the UI component for inputting the question
question = st.text_input("Enter your question:", "Example: Is there a correlation between pre-pregnancy BMI and birth weight?")

def BQ_QA(question, max_fixes=10):
    # Fetch schema columns for BigQuery
    BQ_PROJECT = 'bigquery-public-data'
    BQ_DATASET = 'sdoh_cdc_wonder_natality'
    BQ_TABLES = ['county_natality', 'county_natality_by_mother_race', 'county_natality_by_father_race']
    query = f"""
        SELECT * EXCEPT(field_path, collation_name, rounding_mode)
        FROM `{BQ_PROJECT}.{BQ_DATASET}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
        WHERE table_name in ({','.join([f'"{table}"' for table in BQ_TABLES])})
    """
    schema_columns = bq.query(query=query).to_dataframe()

    # Generate the initial query
    initial_query_text = initial_query(question, schema_columns)
    st.text_area("Generated Query:", initial_query_text, height=100)

    # Run the query and handle errors if any
    query_job = bq.query(query=initial_query_text)
    if query_job.errors:
        st.error('Errors found in the query. Attempting to fix...')
        fixed_query, query_job, fix_tries, codechat = fix_query(initial_query_text, max_fixes)
        if query_job.errors:
            st.error(f'No answer generated after {fix_tries} attempts.')
            # Optionally display the chat history for debugging
            # for message in codechat.get_chat_history():
            #     st.text(message)
        else:
            st.success('Query fixed and executed successfully.')
            results = query_job.to_dataframe()
            st.dataframe(results)
    else:
        st.success('Query executed successfully.')
        results = query_job.to_dataframe()
        st.dataframe(results)

    # Generate and display insights from the query results if there are no errors
    if not query_job.errors:
        question_response = answer_question(question, query_job)
        st.markdown("## Insights")
        st.write(question_response)

# Button to trigger the BQ_QA process
if st.button('Analyze'):
    BQ_QA(question)