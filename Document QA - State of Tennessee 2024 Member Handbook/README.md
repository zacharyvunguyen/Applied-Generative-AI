

# Project Introduction: Document Q&A for Healthcare Benefits Member Handbook

## Overview

The healthcare benefits landscape is continually evolving, making it imperative for members to have access to precise, tailored information. This project introduces an innovative Document Q&A system designed to transform member interaction with their healthcare benefits handbooks. By allowing users to pose specific questions, the system delivers exact answers sourced directly from the official Member Handbooks, including the Cigna Member Handbook for 2024 and the 2024 HCA Healthcare Benefits Brochure.
![question.png](Document%20QA%20-%202024%20HCA%20Healthcare%20Benefit%2Fimg%2Fquestion.png)
![Source.png](Document%20QA%20-%202024%20HCA%20Healthcare%20Benefit%2Fimg%2FSource.png)
![img.png](Document%20QA%20-%202024%20HCA%20Healthcare%20Benefit%2Fimg%2Fimg.png)
![img_2.png](Document%20QA%20-%202024%20HCA%20Healthcare%20Benefit%2Fimg%2Fimg_2.png)
![img_1.png](Document%20QA%20-%202024%20HCA%20Healthcare%20Benefit%2Fimg%2Fimg_1.png)
## Objectives

- **Provide Instant Access:** Members can swiftly uncover details about their healthcare benefits without sifting through extensive documents.
- **Ensure Accuracy and Relevance:** Answers are drawn from the most recent editions of the healthcare benefits handbooks, ensuring current and precise information.
- **Enhance User Experience:** Utilizing advanced AI for information retrieval significantly improves the user experience, making benefits information more accessible.


This project primarily utilizes two sources:
- The [Cigna Member Handbook for 2024](https://www.tn.gov/content/dam/tn/partnersforhealth/documents/cigna_member_handbook_2024.pdf), outlining comprehensive healthcare benefits.
- The [2024 HCA Healthcare Benefits Brochure](https://careers.hcahealthcare.com/system/production/assets/421507/original/2024_HCA_Healthcare_Benefits_Brochure.pdf), providing detailed insights into the benefits offered to HCA Healthcare members.

## Technologies and Tools

- **Vertex AI LLM Embedding and Language Model APIs:** For embedding generation and text processing.
- **Google Cloud Document AI:** To digitize and structure handbook content for efficient querying.
- **Embedding Search Technologies:** Using ScaNN, chromadb, and Vertex AI Matching Engine for relevant document section retrieval.

## Methodology

1. **Document Processing:** Both handbooks are converted into searchable documents, each representing a section.
2. **Embedding Generation:** Develop embeddings for documents and user queries to grasp context and meaning.
3. **Query Processing:** Employ vector similarity searches for the most relevant documents based on query embeddings.
4. **Information Retrieval and Response Generation:** Information from pertinent documents is extracted and synthesized to craft coherent responses, with references for further reading.

## Unique Approach

Our system distinguishes itself by applying the latest in AI and machine learning for personalized information retrieval. It transcends traditional methods by ensuring responses are always derived from the most current handbook editions, significantly elevating the precision and relevance of the information provided to members.

## Prerequisites and Notes

A focus on Google Cloud's AI and machine learning capabilities is essential for this project. Familiarity with Vertex AI, Google Cloud Document AI, and embedding search technologies is required, emphasizing the importance of accessing the latest documents to furnish members with accurate information.

---

This revision accentuates the project's comprehensive approach to providing members with accessible, precise healthcare benefits information, supported by up-to-date resources from both Cigna and HCA Healthcare.