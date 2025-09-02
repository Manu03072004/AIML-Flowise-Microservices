import nest_asyncio
from fastapi import FastAPI, UploadFile, File
from app.services import summarize_text, answer_question, process_pdf, query_pdf, generate_learning_path

nest_asyncio.apply()
app = FastAPI(title="AI Microservices with Flowise + LangChain")

@app.post("/summarize")
async def summarize(text: str):
    return {"summary": summarize_text(text)}

@app.post("/qa")
async def qa(question: str):
    return {"answer": answer_question(question)}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    process_pdf(file_path)
    return {"message": f"{file.filename} uploaded and processed"}

@app.post("/query_pdf")
async def query(query: str):
    return {"response": query_pdf(query)}

@app.post("/learning_path")
async def learning_path(topic: str):
    return {"path": generate_learning_path(topic)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

