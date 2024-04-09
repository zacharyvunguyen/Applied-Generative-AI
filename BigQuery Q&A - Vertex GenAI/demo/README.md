
# BigQuery Insights Explorer
![newapp.gif](imgs%2Fnewapp.gif)
The BigQuery Insights Explorer is an innovative application designed to bridge the gap between complex data queries and intuitive natural language processing (NLP). Utilizing the potent capabilities of Google Cloud's BigQuery for data storage and querying, along with Vertex AI's advanced machine learning models, this application empowers users to extract meaningful insights from vast datasets using simple, natural language questions.

## Purpose of the App

The primary goal of the BigQuery Insights Explorer is to democratize data analysis, making it accessible to users without extensive expertise in SQL or data science. By allowing users to input questions in natural language, the app abstracts the complexity of formulating SQL queries and interpreting raw data outputs. It's particularly beneficial for quick data explorations, hypothesis testing, or gaining insights from data without delving into the technical nuances of database querying languages.

## Potential of the App

The app's potential is vast, ranging from business intelligence and market research to academic research and beyond. By harnessing the power of BigQuery, it can handle petabytes of data, making it suitable for analyzing extensive datasets like user interactions, financial transactions, or scientific data. Furthermore, the integration with Vertex AI models opens up possibilities for automated data analysis, error correction in query formulations, and generating sophisticated insights that would typically require a data analyst or scientist.
![app.png](imgs%2Fapp.png)
## Technologies and Tools Used

- **Streamlit**: For creating a user-friendly web interface for the application.
- **Google Cloud BigQuery**: Utilized for data storage and executing SQL queries.
- **Vertex AI**: Employs code generation and generative models for translating natural language questions into SQL queries, fixing queries, and generating insights.
- **Python**: The core programming language used to build the application.
- **Environment Variables**: Used for configuring the Google Cloud credentials and TensorFlow logging level.

## How to Use the App

1. **Input Your Question**: Type a question about your data in the provided text input field related to the datasets you have access to in BigQuery.
2. **Analysis and Query Generation**: Submit your question, and the app uses a code generation model from Vertex AI to translate your question into a SQL query tailored for BigQuery.
3. **Error Handling and Correction**: If the generated query contains errors or is not optimized, the app iteratively fixes the issues, ensuring the final query is both valid and efficient.
4. **Insight Generation**: With a successful query, the app analyzes the query results and presents insights in an easily understandable format.
5. **Review Results**: Review the generated query, any corrections made, and the final insights directly within the app's interface.

## Sample Questions

1. How has the average pre-pregnancy BMI changed in Davidson County, TN?
2. What is the annual trend in birth rates for Davidson, TN?
3. Average baby weights across counties in Tennessee?
4. What has been the trend in the number of births in Davidson County, TN, from year to year? Is there an increase or decrease?
5. Is there a correlation between the average birth weight and the average number of prenatal visits in Davidson County, TN?
6. What is the most common race of mothers giving birth in Davidson County, TN, according to the "Mothers_Single_Race" field?
7. How do the average weights of newborns vary by county in Tennessee?



## Acknowledgments

- Google Cloud Platform and Vertex AI for providing the tools and services used in this project.
- The Streamlit community for support and inspiration.

This app stands as a testament to the innovative application of AI in the realm of data analysis, simplifying complex processes and making data-driven insights more accessible to a broader audience.
