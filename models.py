from pydantic import BaseModel

class SummarizeInput(BaseModel):
    text: str

class QAInput(BaseModel):
    context: str
    question: str

class LearningPathInput(BaseModel):
    skills: str

class PDFQAInput(BaseModel):
    question: str


