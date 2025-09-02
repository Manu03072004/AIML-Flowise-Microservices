from app.models import llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader

# Store embeddings
vectorstore = None

def summarize_text(text: str) -> str:
    return llm.predict(f"Summarize this text in 3-4 sentences:\n\n{text}")

def answer_question(question: str) -> str:
    return llm.predict(question)

def process_pdf(pdf_path: str):
    global vectorstore
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(docs, embeddings)

def query_pdf(query: str) -> str:
    if not vectorstore:
        return "No PDF uploaded yet."
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)

def generate_learning_path(topic: str) -> str:
    return llm.predict(f"Create a step-by-step learning path for: {topic}")

