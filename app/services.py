
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

# HuggingFace pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
text_generator = pipeline("text-generation", model="gpt2")

# Global storage for uploaded PDFs
pdf_texts = []
faiss_db = None

def process_pdf(file):
    global pdf_texts, faiss_db
    reader = PyPDF2.PdfReader(file)
    text = "".join([page.extract_text() for page in reader.pages])
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_db = FAISS.from_texts(chunks, embeddings)
    
    pdf_texts.append(text)
    return len(chunks)

def query_pdf(question):
    global faiss_db
    if faiss_db is None:
        return None, "No PDF uploaded yet."
    docs = faiss_db.similarity_search(question, k=3)
    context = " ".join([d.page_content for d in docs])
    answer = qa_pipeline(question=question, context=context)
    return answer["answer"], context

