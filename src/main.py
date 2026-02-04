from fastapi import FastAPI, HTTPException
from src.services.openai_service import summarize_text, analyze_sentiment
from src.config.settings import settings

app = FastAPI(title="Text Processing API")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/summarize")
def summarize(payload: dict):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    summary = summarize_text(text)
    return {"summary": summary, "original_length": len(text), "summary_length": len(summary)}

@app.post("/analyze-sentiment")
def sentiment(payload: dict):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    return analyze_sentiment(text)


