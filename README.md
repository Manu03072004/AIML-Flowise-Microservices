# AIML-Flowise-Microservices

## Overview
A modular AI microservices project using **Flowise** and **LangChain**.  
This project provides APIs for:
- Text Summarization  
- Q&A over documents (PDFs)  
- Dynamic Learning Path suggestion  

It uses **OpenRouter** LLM and **RAG** (Retrieval-Augmented Generation) for document-based question answering.

---

## Features
- **Text Summarization:** Summarizes long text into concise summaries.  
- **Q&A over Documents:** Upload PDFs and query for answers.  
- **Learning Path Generator:** Generates personalized learning paths based on given skills.  
- **RAG Integration:** Uses vector stores (FAISS) for efficient document retrieval.  
- **OpenRouter LLM:** Lightweight, small models for faster inference.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Manu03072004/AIML-Flowise-Microservices.git
cd AIML-Flowise-Microservices
```

Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file in the project root:

OPENROUTER_API_KEY=your_api_key_here

2. Ensure you have your OpenRouter API Key.

## Running the Project Locally

1. Make sure your virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```
2. Run the FastAPI server:
   ```bash
   python main.py
   ```
3.Open your browser and access the API docs (Swagger UI):

API will be accessible at:
http://127.0.0.1:8000
Swagger UI:
http://127.0.0.1:8000/docs

---

## Author

**Manaswini Pusarla**
