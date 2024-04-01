# Project Introduction: Document Q&A for Healthcare Benefits Member Handbook

## Overview

In the evolving landscape of healthcare benefits, access to accurate and personalized information is crucial for members. Our project aims to revolutionize how members interact with their healthcare benefits handbooks by implementing a cutting-edge Document Q&A system. This system allows users to ask specific questions about their healthcare benefits and receive precise answers drawn directly from the official Member Handbook.

## Objectives

- **Provide Instant Access:** Enable members to quickly find answers to their questions regarding healthcare benefits without navigating through dense documents.
- **Ensure Accuracy and Relevance:** By sourcing answers directly from the latest version of the Cigna Member Handbook for 2024, we ensure that the information provided is accurate and up-to-date.
- **Enhance User Experience:** Improve the overall member experience with the healthcare benefits provider by leveraging advanced AI technologies for information retrieval.

## Sources

The primary source of information for this project is the [Cigna Member Handbook for 2024](https://www.tn.gov/content/dam/tn/partnersforhealth/documents/cigna_member_handbook_2024.pdf), a comprehensive document outlining the healthcare benefits available to members.

## Technologies and Tools

- **Vertex AI LLM Embedding and Language Model APIs:** Utilize Google Cloud's Vertex AI for embedding generation and text processing to understand and generate human-like responses.
- **Google Cloud Document AI:** Process and convert the Member Handbook into structured documents that can be efficiently queried.
- **Embedding Search Technologies:** Implement embedding search with tools such as ScaNN, chromadb, and Vertex AI Matching Engine to find the most relevant document sections in response to user queries.

## Methodology

1. **Document Processing:** Transform the Member Handbook into searchable documents, with each document representing a section of the handbook.
2. **Embedding Generation:** Create embeddings for both the documents and user queries to understand their context and meaning.
3. **Query Processing:** Use vector similarity searches to identify the most relevant documents based on the query embeddings.
4. **Information Retrieval and Response Generation:** Extract and synthesize information from the identified documents to generate coherent and accurate responses to user queries, accompanied by references to the specific sections of the handbook for further reading.

## Unique Approach

Our system stands out by leveraging the latest in AI and machine learning to offer personalized and precise information retrieval. Unlike traditional methods, which may rely on general knowledge that can become outdated, our approach ensures that responses are always based on the current version of the Member Handbook. This project not only enhances the accessibility of information but also significantly improves the accuracy and relevance of the responses provided to members.

## Prerequisites and Notes

This project is designed with a focus on leveraging Google Cloud's AI and machine learning capabilities, and it requires familiarity with Vertex AI, Google Cloud Document AI, and embedding search technologies. Our methodology underscores the importance of direct access to the most current documents, ensuring that members receive the most accurate information possible.