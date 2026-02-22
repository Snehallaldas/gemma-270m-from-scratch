from fastapi import FastAPI
from pydantic import BaseModel
from .inference import generate

app = FastAPI()

class Request(BaseModel):
    text: str
    max_tokens: int = 100

@app.post("/generate")
def generate_text(req: Request):
    result = generate(req.text, req.max_tokens)
    return {"response": result}