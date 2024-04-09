import os
from app_without_precheck import document_bot

# Define the path to your service account key file
service_account_key_path = "/Users/zacharynguyen/Documents/GitHub/2024/Applied-Generative-AI/IAM/zacharynguyen-genai-656c475b142a.json"
# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path

question ="What is the process for filing an appeal if a claim is denied?"
#prompt = document_bot(question, display_contexts = True, display_annotations = True)
prompt = document_bot(question, max_output_tokens=1000, DISTANCE=0.2, MODEL='GEMINI', display_contexts=True,
                 display_annotations=True, ground=True)