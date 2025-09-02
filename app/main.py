
from fastapi import FastAPI, UploadFile, File
from app.models import SummarizeInput, QAInput, LearningPathInput, PDFQAInput
from app.services import summarizer, qa_pipeline, text_generator, process_pdf, query_pdf

app = FastAPI(title="AI Microservices with RAG & HuggingFace")

# Summarization
@app.post("/summarize")
def summarize(data: SummarizeInput):
    summary = summarizer(data.text, max_length=100, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

# Q&A
@app.post("/qa")
def qa(data: QAInput):
    answer = qa_pipeline(question=data.question, context=data.context)
    return {"answer": answer["answer"]}

# Learning Path
@app.post("/learning-path")
def learning_path(data: LearningPathInput):
    prompt = f"Suggest a personalized learning path for someone with these skills: {data.skills}"
    path = text_generator(prompt, max_length=150, num_return_sequences=1)
    return {"learning_path": path[0]["generated_text"]}

# PDF upload for RAG
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    chunks_count = process_pdf(file.file)
    return {"message": f"PDF uploaded and indexed with {chunks_count} chunks."}

# PDF Q&A using RAG
@app.post("/qa-pdf")
def qa_pdf(data: PDFQAInput):
    answer, context = query_pdf(data.question)
    if answer is None:
        return {"error": context}
    return {"answer": answer, "context": context}

