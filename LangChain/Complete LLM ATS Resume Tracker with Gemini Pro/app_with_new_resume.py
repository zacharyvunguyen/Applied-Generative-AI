import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure generative AI model with API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to interact with generative AI
def get_gemini_response(input_text):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(input_text)
        return response.text
    except Exception as e:
        st.error("Error generating response: " + str(e))
        return None

# Function to extract text from uploaded PDF
def extract_pdf_text(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle potential None values
        return text
    except Exception as e:
        st.error("Error reading PDF: " + str(e))
        return None

# Prompt template for generative AI
input_prompt_template = """
Act as a skilled ATS expert in tech fields like software engineering, data science, data analytics, and big data engineering.
Evaluate the resume based on the given job description, considering the competitive job market. Provide a percentage match with the job description,
identify missing keywords, and suggest an enhanced resume.
Resume: {text}
Job Description: {jd}
Output the result as a JSON-formatted string with the following structure:
{{"JD Match": "%", "Missing Keywords": [], "Enhanced Resume": ""}}
"""

# Streamlit app design
st.title("Smart ATS")
st.markdown("### Improve Your Resume for ATS")
st.markdown(
    "Upload your resume and paste a job description. "
    "The AI system will analyze your resume, provide a matching percentage, "
    "identify missing keywords, and suggest enhancements."
)

# Input fields for job description and resume file
jd = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader(
    "Upload Your Resume", type="pdf", help="Please upload a PDF"
)

submit = st.button("Submit")

if submit:
    if jd and uploaded_file:
        resume_text = extract_pdf_text(uploaded_file)
        if resume_text:
            input_prompt = input_prompt_template.format(text=resume_text, jd=jd)
            response = get_gemini_response(input_prompt)

            if response and response.strip():
                # Debug response content
                st.markdown("#### Raw AI Response")
                st.write(response.strip())  # Display raw response for debugging

                try:
                    # Attempt to parse JSON
                    result_data = json.loads(response.strip())
                    st.subheader("ATS Analysis Results")
                    st.markdown("#### Job Description Match")
                    st.write(result_data.get("JD Match", "N/A"))

                    st.markdown("#### Missing Keywords")
                    missing_keywords = result_data.get("Missing Keywords", [])
                    if missing_keywords:
                        st.write(", ".join(missing_keywords))
                    else:
                        st.write("No missing keywords found.")

                    st.markdown("#### Enhanced Resume Suggestions")
                    st.write(result_data.get("Enhanced Resume", "No suggestions available."))
                except json.JSONDecodeError:
                    # Handle JSON parsing error
                    st.error("Could not parse response. The AI system may have returned an unexpected format.")
            else:
                st.warning("Received an empty or invalid response. Please try again.")
        else:
            st.warning("Could not extract text from the PDF. Please check your file.")
    else:
        st.warning("Please provide a job description and upload a PDF resume.")
