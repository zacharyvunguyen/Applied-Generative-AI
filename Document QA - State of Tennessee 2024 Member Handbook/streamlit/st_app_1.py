import streamlit as st
import os

# Set up the path to your Google Cloud service account key file
service_account_key_path = "/Users/zacharynguyen/Documents/GitHub/2024/Applied-Generative-AI/IAM/zacharynguyen-genai-656c475b142a.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path

# Import the document_bot function from your application
from app_without_precheck import document_bot


# Streamlit app starts here
def main():
    st.title("Document Bot Query Interface")

    # User inputs the question
    question = st.text_input("Enter your question:", "What is the process for filing an appeal if a claim is denied?")

    if st.button("Generate Prompt"):
        # Call document_bot with the user's question
        prompt = document_bot(question, max_output_tokens=1000, DISTANCE=0.2, MODEL='GEMINI',
                              display_contexts=True, display_annotations=True, ground=True)

        # Display the question and the generated prompt
        st.write("### Question:")
        st.write(question)
        st.write("### Generated Prompt:")
        st.write(prompt)


if __name__ == "__main__":
    main()
