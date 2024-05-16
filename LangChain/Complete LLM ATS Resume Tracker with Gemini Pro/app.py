import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json

load_dotenv()  # Load all our environment variables

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY is not set in the environment variables.")
    st.stop()
genai.configure(api_key=api_key)

def get_gemini_response(prompt):
    """Generates content using Gemini Pro model based on the provided prompt.
    Handles potential exceptions that could arise during the API call."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating content: {str(e)}")
        return None

def extract_text_from_pdf(uploaded_file):
    """Extracts text from all pages of the uploaded PDF file."""
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text() or " "
            text += extracted_text
        return text
    except Exception as e:
        st.error(f"Failed to read PDF file: {str(e)}")
        return ""

# Adjust the prompt to use single braces around the placeholders
input_prompt = """
As an advanced and experienced Applicant Tracking System (ATS) with a specialized focus on the technology sector—including 
software engineering, data science, data analysis, and big data engineering—your task is to meticulously compare the given resume 
with the provided job description. 
Because the job market is highly competitive, it's crucial to provide thorough and detailed feedback.
Your objective is to analyze the resume for its alignment with the job description, identify critical keywords that are missing, 
and suggest specific modifications to ensure the resume matches the job requirements. 
When offering recommendations, use the original content as a foundation, rephrasing experience or projects as needed, 
without removing or adding new ones.

Resume content: {text}
Job description: {jd}

Expected output format:
{{"JD Match": "%", "Missing Keywords": [], "Suggested Resume": "", "JD Match with new Resume": "%"}}
"""

## Streamlit app setup
st.title("Smart ATS")
st.subheader("Improve Your Resume for ATS")

jd = st.text_area("Paste the Job Description here:",)
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload your resume in PDF format.")

if st.button("Submit"):
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        full_prompt = input_prompt.format(text=text, jd=jd)  # Format the prompt with user inputs
        response = get_gemini_response(full_prompt)

        if response:
            try:
                response_data = json.loads(response)  # Parse the JSON response
                st.markdown("### Analysis Result")
                st.markdown(f"**Job Description Match**: `{response_data['JD Match']}`")
                st.markdown("**Missing Keywords:**")
                st.markdown("- " + "\n- ".join(response_data["Missing Keywords"]))  # List missing keywords as bullet points
                st.markdown("**Suggested Resume Enhancements**")
                st.markdown(response_data["Suggested Resume"])
                st.markdown(f"**New Job Description Match**: `{response_data['JD Match with new Resume']}`%")
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse model response: {str(e)}")
        else:
            st.error("Failed to get a valid response from the model.")
    else:
        st.warning("Please upload a resume in PDF format.")
