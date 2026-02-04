from openai import OpenAI
from fastapi import HTTPException
from src.config.settings import settings

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

SUMMARIZE_PROMPT = """Summarize the following text briefly:

Text: {text}

Summary:"""

SENTIMENT_PROMPT = """Analyze the sentiment of the following text and classify it as positive, negative, or neutral:

Text: {text}

Respond as: Sentiment: [positive/negative/neutral]
"""

def call_openai(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

def summarize_text(text: str) -> str:
    prompt = SUMMARIZE_PROMPT.format(text=text)
    return call_openai(prompt)

def analyze_sentiment(text: str) -> dict:
    prompt = SENTIMENT_PROMPT.format(text=text)
    response = call_openai(prompt)

    # Simple parsing: detect positive/negative/neutral
    sentiment = "neutral"
    for word in ["positive", "negative", "neutral"]:
        if word in response.lower():
            sentiment = word
            break
    return {"sentiment": sentiment, "raw_response": response}

