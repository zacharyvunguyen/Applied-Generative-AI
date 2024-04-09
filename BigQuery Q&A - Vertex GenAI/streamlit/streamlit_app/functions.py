import os
# Import Google Cloud and Vertex AI libraries
from google.cloud import bigquery
import vertexai
import vertexai.language_models
import vertexai.preview.generative_models


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

##FUNCTIONS
def initial_query(question, schema_columns):
    # code generation model
    codegen_model = vertexai.language_models.CodeGenerationModel.from_pretrained('code-bison@002')

    # initial request for query:
    context_prompt = f"""
context (BigQuery Table Schema):
{schema_columns.to_markdown(index=False)}

Write a query for Google BigQuery using fully qualified table names to answer this question:
{question}
"""

    context_query = codegen_model.predict(context_prompt, max_output_tokens=256)

    # extract query from response
    if context_query.text.find("```") >= 0:
        context_query = context_query.text.split("```")[1]
        if context_query.startswith('sql'):
            context_query = context_query[3:]
        print('Initial Query:\n', context_query)
    else:
        print(
            'No query provided (first try) - unforseen error, printing out response to help with editing this funcntion:\n',
            query_response.text)

    return context_query

def codechat_start(question, query, schema_columns):
    # code chat model
    codechat_model = vertexai.language_models.CodeChatModel.from_pretrained('codechat-bison@002')

    # start a code chat session and give the schema for columns as the starting context:
    codechat = codechat_model.start_chat(
        context=f"""
The BigQuery Environment has tables defined by the follow schema:
{schema_columns.to_markdown(index=False)}

This session is trying to troubleshoot a Google BigQuery SQL query that is being writen to answer the question:
{question}

BigQuery SQL query that needs to be fixed:
{query}

Instructions:
As the user provides versions of the query and the errors returned by BigQuery, offer suggestions that fix the errors but it is important that the query still answer the original question.
"""
    )

    return codechat

def fix_query(query, max_fixes):
    # iteratively run query, and fix it using codechat until success (or max_fixes reached):
    fix_tries = 0
    answer = False
    while fix_tries < max_fixes:
        if not query:
            return
        # run query:
        query_job = bq.query(query=query)
        # if errors, then generate repair query:
        if query_job.errors:
            fix_tries += 1

            if fix_tries == 1:
                codechat = codechat_start(question, query, schema_columns)

            # construct hint from error
            hint = ''
            for error in query_job.errors:
                # detect error message
                if 'message' in list(error.keys()):
                    # detect index of error location
                    if error['message'].rindex('[') and error['message'].rindex(']'):
                        begin = error['message'].rindex('[') + 1
                        end = error['message'].rindex(']')
                        # verify that it looks like an error location:
                        if end > begin and error['message'][begin:end].index(':'):
                            # retrieve the two parts of the error index: query line, query column
                            query_index = [int(q) for q in error['message'][begin:end].split(':')]
                            hint += query.split('\n')[query_index[0] - 1].strip()
                            break

            # construct prompt to request a fix:
            fix_prompt = f"""This query:\n{query}\n\nReturns these errors:\n{query_job.errors}"""

            if hint != '':
                fix_prompt += f"""\n\nHint, the error appears to be in this line of the query:\n{hint}"""

            query_response = codechat.send_message(fix_prompt)
            query_response = codechat.send_message(
                'Respond with only the corrected query that still answers the question as a markdown code block.')
            if query_response.text.find("```") >= 0:
                query = query_response.text.split("```")[1]
                if query.startswith('sql'):
                    query = query[4:]
                print(f'Fix #{fix_tries}:\n', query)
            # response did not have a query????:
            else:
                query = ''
                print('No query in response...')

        # no error, break while loop
        else:
            break

    return query, query_job, fix_tries, codechat

def answer_question(question, query_job):
    # text generation model
    gemini_model = vertexai.preview.generative_models.GenerativeModel("gemini-pro")

    # answer question
    result = query_job.to_dataframe()
    question_prompt = f"""
Please derive insights from: {question}
Utilize the statistics from the BigQuery table relevant to this inquiry. Emphasize crucial discoveries and their implications for strategic actions, ensuring to mention specific statistics where possible. Avoid repeating the question or detailing the dataset's context.

Use this data:
{result.to_markdown(index=False)}
    """

    question_response = gemini_model.generate_content(question_prompt)

    return question_response.text
