🤖 AI Chatbot Project
Traditional NLP vs Retrieval-Augmented Generation (RAG)
📌 Project Overview

This project implements and compares two different chatbot architectures:

Traditional NLP-based Chatbot using TF-IDF + Cosine Similarity

Modern Retrieval-Augmented Generation (RAG) Chatbot using

Sentence Embeddings

FAISS Vector Database

Large Language Model (FLAN-T5)

The goal of the project is to demonstrate how modern RAG-based systems provide more accurate, scalable, and hallucination-free responses compared to traditional similarity-based chatbots.

🧠 Motivation

Traditional chatbots rely on exact or near-exact text similarity, which limits their ability to:

Understand paraphrased queries

Generate new responses

Scale to large or diverse knowledge bases

RAG overcomes these limitations by:

Retrieving relevant knowledge semantically

Using a generative model to synthesize answers

Preventing hallucination by grounding responses in retrieved context

🧩 Project Architecture
1️⃣ Traditional Chatbot Pipeline (TF-IDF)
User Input
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Cosine Similarity
   ↓
Most Similar Stored Question
   ↓
Return Predefined Response

2️⃣ RAG Chatbot Pipeline
User Input (Dynamic)
   ↓
Text Cleaning
   ↓
Sentence Embedding
   ↓
FAISS Vector Search
   ↓
Retrieve Relevant Chunks
   ↓
Augment Prompt with Context
   ↓
LLM (FLAN-T5) Generation
   ↓
Final Answer

📂 Datasets Used
🔹 1. Conversational Dataset (Custom)

Contains human-to-human dialog pairs

Used for:

Greetings

Small talk

Casual conversation

Stored as a text file or CSV

🔹 2. BioASQ Dataset (rag-mini-bioasq)

Biomedical knowledge passages

Used for:

Domain-specific question answering

Loaded from Hugging Face:

rag-datasets/rag-mini-bioasq

⚙️ Technologies & Libraries

Python

scikit-learn (TF-IDF, cosine similarity)

Sentence-Transformers (semantic embeddings)

FAISS (vector database)

Hugging Face Transformers

FLAN-T5-Large (generative model)

Google Colab (development environment)

🧪 Approach 1: Traditional Cosine Similarity Chatbot
🔹 Methodology

Text is cleaned and normalized

User inputs and dataset questions are converted into TF-IDF vectors

Cosine similarity is used to find the closest match

The response corresponding to the most similar question is returned

🔹 Advantages

Simple to implement

Fast for small datasets

No GPU required

🔹 Limitations

Cannot understand paraphrases

Returns only existing responses

No reasoning or generation

Poor scalability

🧠 Approach 2: Retrieval-Augmented Generation (RAG)
🔹 Methodology (Step-by-Step)

Collect Knowledge Sources

BioASQ biomedical passages

Custom dialog text files

Clean & Prepare Data

Lowercasing

Removing noise

Normalization

Chunk the Data

BioASQ: Already chunked (passage-level)

Dialog data: Split into conversation pairs

Convert Chunks into Embeddings

SentenceTransformer (all-MiniLM-L6-v2)

Captures semantic meaning

Store Embeddings in FAISS

Enables fast similarity search

User Asks a Question (Dynamic Input)

Retrieve Relevant Chunks

Semantic similarity search in FAISS

Augment Prompt with Context

Inject retrieved chunks into LLM prompt

Strict instruction to avoid hallucination

Generate Answer

Using google/flan-t5-large

Return Final Answer

Safe, grounded, and contextual

🧪 Testing & Evaluation
✔ Tested Scenarios

Biomedical questions → Correct factual answers

Greetings → Conversational replies

Unknown questions → Safe refusal

✔ Example
User: What is Hirschsprung disease?
Bot: Hirschsprung disease involves mutations in several genes and is considered a multifactorial disorder.

User: What is the dental policy?
Bot: I don't have that information.

📊 Comparison of Both Approaches
Feature	                   TF-IDF Chatbot	   RAG Chatbot
Semantic Understanding	       ❌	              ✅
Handles Paraphrases	           ❌	            ✅
Generates New Text	           ❌	            ✅
Hallucination Control	         ❌	              ✅
Multi-Source Knowledge	       ❌	              ✅
Scalability	                   Low	            High

🏆 Key Learnings

TF-IDF works only for simple similarity-based systems

RAG is the industry standard for knowledge-based chatbots

Separating retrieval and generation greatly improves accuracy

Prompt constraints are critical to prevent hallucination

🚀 Future Enhancements

Add citation support (source-aware answers)

Improve retrieval with hybrid search

Deploy as a web or mobile application

Fine-tune embedding models for specific domains

📌 Conclusion

This project demonstrates the evolution of chatbot systems from traditional NLP techniques to modern Retrieval-Augmented Generation architectures.
The RAG-based chatbot significantly outperforms the cosine similarity chatbot in terms of accuracy, flexibility, and safety, making it suitable for real-world applications such as enterprise assistants, medical QA systems, and knowledge bots.

👨‍💻 Author

Project Type: Deep Learning / NLP
Approach: Traditional NLP + RAG
Environment: Google Colab
