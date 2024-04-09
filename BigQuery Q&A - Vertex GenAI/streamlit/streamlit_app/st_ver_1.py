import streamlit as st
import os
# Import Google Cloud and Vertex AI libraries
from google.cloud import bigquery
import vertexai
import vertexai.language_models
import vertexai.preview.generative_models
#from functions import initial_query,codechat_start,fix_query,answer_question

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
BQ_TABLES = ['county_natality', 'county_natality_by_mother_race']

# Construct and execute a BigQuery query to retrieve schema columns
query = f"""
    SELECT * EXCEPT(field_path, collation_name, rounding_mode)
    FROM `{BQ_PROJECT}.{BQ_DATASET}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
    WHERE table_name in ({','.join([f'"{table}"' for table in BQ_TABLES])})
"""
schema_columns = bq.query(query=query).to_dataframe()
question = "Is there a seasonal pattern to the number of births in 2018?"

#
def initial_query(question, schema_columns):
    # code generation model
    codegen_model = vertexai.language_models.CodeGenerationModel.from_pretrained('code-bison@002')

    # initial request for query:
    context_prompt = f"""
context (BigQuery Table Schema):
{schema_columns.to_markdown(index=False)}

Write a query for Google BigQuery using fully qualified table names to answer this question:
{question}
Note: Specifically, when filtering data by county, use 'County_of_Residence_FIPS' rather than 'County_of_Residence' to ensure precision and accuracy. 
For example, avoid using WHERE clauses like `County_of_Residence = 'Davidson County'` and instead use `County_of_Residence_FIPS = '47037'` for filtering."
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
"As the user provides versions of the query and the errors returned by BigQuery, offer suggestions that correct these errors, ensuring each revised query accurately addresses the original question. 
Specifically, when filtering data by county, use 'County_of_Residence_FIPS' rather than 'County_of_Residence' to ensure precision and accuracy. For example, avoid using WHERE clauses like `County_of_Residence = 'Davidson County'` and instead use `County_of_Residence_FIPS = '47037'` for filtering."

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
"Analyze the dataset in relation to: '{question}'. 
Identify critical statistical insights within the BigQuery table data. 
Utilize these insights to construct detailed, actionable recommendations. 
Your analysis should directly address the core of the question, focusing on specific findings 
that can guide data-driven decision-making. Avoid broad summaries or restating the question. 
Highlight how the data supports each insight, providing a clear path from analysis to action for a data scientist or analyst."



Use this data:
{result.to_markdown(index=False)}
    """

    question_response = gemini_model.generate_content(question_prompt)

    return question_response.text


# Sidebar for app navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a view",
                            ["Query Analysis", "App Description", ])
st.sidebar.markdown("""
## BigQuery Configuration
**Project:** `bigquery-public-data`  
**Dataset:** `sdoh_cdc_wonder_natality`  
**Tables:**  
- `county_natality`  
- `county_natality_by_mother_race`    
""")

if app_mode == "App Description":
    st.title("BigQuery Insights Explorer 🚀")
    st.markdown("""
    The app, ingeniously designed to bridge the gap between complex data queries and intuitive natural language processing (NLP) 🧠, revolutionizes the way we interact with large datasets. 
    At its core, it utilizes the potent capabilities of Google Cloud's BigQuery 🌐 for data storage and querying, alongside Vertex AI's advanced machine learning models 🤖. 
    This blend of technologies empowers users to extract meaningful insights from vast datasets using simple, natural language questions 🗣️.

    ### Purpose of the App 🎯

    The primary purpose of this app is to democratize data analysis, making it accessible to users without extensive expertise in SQL or data science 🔍. 
    By allowing users to input questions in natural language, the app abstracts the complexity of formulating SQL queries and interpreting raw data outputs. 
    It's particularly beneficial for quick data explorations, hypothesis testing, or gaining insights from data without the need to dive into the technical nuances of database querying languages.

    ### Potential of the App 🔥

    The app's potential is vast, ranging from business intelligence and market research to academic research and beyond 📊. 
    By harnessing the power of BigQuery, it can handle petabytes of data, making it suitable for analyzing extensive datasets like user interactions, financial transactions, or scientific data. 
    Furthermore, the integration with Vertex AI models opens up possibilities for automated data analysis, error correction in query formulations, and generating sophisticated insights that would typically require a data analyst or scientist 🧑‍🔬.

    ### How to Use the App 📝

    Using the app is straightforward:

    1. **Input Your Question**: Start by typing a question about your data in the provided text input field. The question should be related to the datasets you have access to in BigQuery and can be as simple or complex as needed.
    2. **Analysis and Query Generation**: Once you submit your question, the app uses a code generation model from Vertex AI to translate your natural language question into a SQL query tailored for BigQuery.
    3. **Error Handling and Correction**: If the generated query contains errors or is not optimized, the app employs a code chat model to iteratively fix the issues, ensuring the final query is both valid and efficient.
    4. **Insight Generation**: With a successful query, the app then leverages another generative AI model to analyze the query results and present insights in an easily understandable format.
    5. **Review Results**: You can review the generated query, any corrections made, and the final insights directly within the app's interface.

    Sample questions:

    1. Which mother's single race category reports the highest average number of births?
    2. How does the average gestational age at birth vary by the mother's single race?
    3. What is the average age of mothers for each mother's single race category?
    4.What is the annual trend in birth rates for Davidson, TN?
    5.Average baby weights across counties in Tennessee?
    6.What has been the trend in the number of births in Davidson County, TN, from year to year? Is there an increase or decrease?
    7.Is there a correlation between the average birth weight and the average number of prenatal visits in Davidson County, TN?
    8.What is the most common race of mothers giving birth in Davidson County, TN, according to the "Mothers_Single_Race" field?
    9.How do the average weights of newborns vary by county in Tennessee?

    This app stands as a testament to the innovative application of AI in the realm of data analysis, simplifying complex processes and making data-driven insights more accessible to a broader audience 🌟.
    """)
elif app_mode == "Query Analysis":
    # UI for Query Analysis
    st.title("BigQuery Analysis and Insights")


    def BQ_QA(question, max_fixes=10):
        with st.spinner('Fetching schema columns from BigQuery...'):
            BQ_PROJECT = 'bigquery-public-data'
            BQ_DATASET = 'sdoh_cdc_wonder_natality'
            BQ_TABLES = ['county_natality', 'county_natality_by_mother_race']
            query = f"""
                SELECT * EXCEPT(field_path, collation_name, rounding_mode)
                FROM `{BQ_PROJECT}.{BQ_DATASET}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
                WHERE table_name in ({','.join([f'"{table}"' for table in BQ_TABLES])})
            """
            bq_client = bigquery.Client()
            schema_columns = bq_client.query(query=query).to_dataframe()

        with st.spinner('Generating initial query...'):
            initial_query_text = initial_query(question, schema_columns)

        query_to_run = initial_query_text  # Assume initial query will be run

        with st.spinner('Executing query...'):
            query_job = bq_client.query(query=initial_query_text)
            if query_job.errors:
                fixed_query, query_job, fix_tries, codechat = fix_query(initial_query_text, max_fixes)
                if query_job.errors:
                    st.error(f'No answer generated after {fix_tries} attempts.')
                    return
                else:
                    st.success('Query fixed and executed successfully.')
                    query_to_run = fixed_query  # Update query_to_run to be the fixed query
            else:
                st.success('Query executed successfully.')

        # Show the query that was successfully executed (either initial or fixed)
        st.subheader("Executed Query")
        st.code(query_to_run, language='sql')

        results = query_job.to_dataframe()
        st.subheader("Query Results")
        st.dataframe(results)

        with st.spinner('Generating insights...'):
            question_response = answer_question(question, query_job)
            st.subheader("Insights")
            st.write(question_response)


    # GUI for question input
    question = st.text_input("Enter your question:", "Is there a seasonal pattern to the number of births in 2018?")
    if st.button('Analyze'):
        BQ_QA(question)