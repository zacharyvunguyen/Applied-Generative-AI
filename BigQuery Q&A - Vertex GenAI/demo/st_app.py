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

# Sidebar for app navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a view",
                            [ "Query Analysis","App Description",])
st.sidebar.markdown("""
## BigQuery Configuration
**Project:** `bigquery-public-data`  
**Dataset:** `sdoh_cdc_wonder_natality`  
**Tables:**  
- `county_natality`  
- `county_natality_by_mother_race`  
- `county_natality_by_father_race`  
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
    This app stands as a testament to the innovative application of AI in the realm of data analysis, simplifying complex processes and making data-driven insights more accessible to a broader audience 🌟.
    """)
elif app_mode == "Query Analysis":
    # UI for Query Analysis
    st.title("Query Analysis")
    question = st.text_input("Enter your question:",
                             "Example: Which mother's single race category reports the highest average number of births??")


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
        #st.text_area("Generated Query:", initial_query_text, height=300)
        st.code(initial_query_text,language='sql')

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


    # Function body as previously defined

    # Trigger the BQ_QA process
    if st.button('Analyze'):
        BQ_QA(question)
