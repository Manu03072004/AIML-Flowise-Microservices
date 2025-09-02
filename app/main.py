import nest_asyncio
import uvicorn
import threading
from pyngrok import ngrok
from fastapi import FastAPI, UploadFile, File

from app.models import SummarizeInput, QAInput, LearningPathInput, PDFQAInput
from app.services import summarizer, qa_pipeline, text_generator, process_pdf, query_pdf

# Patch asyncio for notebooks/Colab
nest_asyncio.apply()

# Initialize FastAPI with repo title
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
# Run FastAPI in background thread
# ------------------------
def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_app)
thread.start()

# ------------------------
# Ngrok public URL (temporary)
# ------------------------
# Replace with your ngrok auth token
!ngrok authtoken 323AJdV2G2TyaCWMJWW6R7Acs7Q_5UmhesJXxjfXKpcL2S82y
public_url = ngrok.connect(8000)
print("Your API is live at:", public_url)
print("Add /docs at the end to open Swagger UI")

