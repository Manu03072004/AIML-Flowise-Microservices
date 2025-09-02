import nest_asyncio
import uvicorn
import threading
from fastapi import FastAPI, UploadFile, File

from app.models import SummarizeInput, QAInput, LearningPathInput, PDFQAInput
from app.services import summarizer, qa_pipeline, text_generator, process_pdf, query_pdf

# Patch asyncio for notebooks (optional)
nest_asyncio.apply()

# Initialize FastAPI
app = FastAPI(title="AIML-Flowise-Microservices")

# ------------------------
# Endpoints
# ------------------------

@app.post("/summarize")
def summarize_text(data: SummarizeInput):
    summary = summarizer(data.text, max_length=100, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

@app.post("/qa")
def question_answer(data: QAInput):
    answer = qa_pipeline(question=data.question, context=data.context)
    return {"answer": answer["answer"]}

@app.post("/learning-path")
def learning_path(data: LearningPathInput):
    prompt = f"Suggest a personalized learning path for someone with these skills: {data.skills}"
    path = text_generator(prompt, max_length=150, num_return_sequences=1)
    return {"learning_path": path[0]["generated_text"]}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    chunks_count = process_pdf(file.file)
    return {"message": f"PDF uploaded and indexed with {chunks_count} chunks."}

@app.post("/qa-pdf")
async def qa_pdf(data: PDFQAInput):
    answer, context = query_pdf(data.question)
    if answer is None:
        return {"error": context}
    return {"answer": answer, "context": context}

# ------------------------
# Run FastAPI
# ------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
