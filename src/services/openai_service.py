import os
from openai import OpenAI
from fastapi import HTTPException

from src.config.settings import settings

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ============================================================================
# Prompt Templates
# ============================================================================

# Summarization prompts
SUMMARIZE_PROMPTS = {
    "single": """Summarize the text below in no more than 4-5 sentences. Focus only on the main points:

{text}

Summary:""",
    "one": """Here is some context and an example:
Example: "The sun rises in the east and sets in the west."
Summary: "Sun rises east, sets west."

Now summarize the following text in 4-5 sentences:

{text}

Summary:""",
    "multi": """You are given multiple pieces of information. Summarize each key idea as a bullet point, then write a final summary in 4-5 sentences:

{text}

Summary:"""
}

# Sentiment analysis prompts
SENTIMENT_PROMPTS = {
    "single": """Analyze the sentiment of the following text. Classify as positive, negative, or neutral. Explain briefly:

{text}

Response:
Sentiment: [positive/negative/neutral]
Explanation:""",
    "one": """Here is an example:
Text: "I love ice cream."
Sentiment: positive
Explanation: expresses enjoyment of ice cream.

Now analyze this text similarly:

{text}

Response:
Sentiment: [positive/negative/neutral]
Explanation:""",
    "multi": """Analyze the sentiment considering multiple aspects: tone, context, and intent. Provide classification and brief explanation:

{text}

Response:
Sentiment: [positive/negative/neutral]
Explanation:"""
}

# ============================================================================
# Helper functions
# ============================================================================

def call_openai(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.3) -> str:
    """Call OpenAI API safely."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


# ============================================================================
# Main Service Functions
# ============================================================================

def summarize_text(text: str, variant: str = "single") -> str:
    """Summarize text using the chosen prompt variant."""
    prompt_template = SUMMARIZE_PROMPTS.get(variant, SUMMARIZE_PROMPTS["single"])
    prompt = prompt_template.format(text=text)
    return call_openai(prompt)


def analyze_sentiment(text: str, variant: str = "single") -> dict:
    """Analyze sentiment using the chosen prompt variant."""
    prompt_template = SENTIMENT_PROMPTS.get(variant, SENTIMENT_PROMPTS["single"])
    prompt = prompt_template.format(text=text)
    response = call_openai(prompt, temperature=0.2)

    # Simple parser for Sentiment: and Explanation:
    sentiment = "neutral"
    explanation = response
    for line in response.split("\n"):
        line_lower = line.lower()
        if "sentiment:" in line_lower:
            if "positive" in line_lower:
                sentiment = "positive"
            elif "negative" in line_lower:
                sentiment = "negative"
            elif "neutral" in line_lower:
                sentiment = "neutral"
        elif "explanation:" in line_lower:
            explanation = line.split(":", 1)[1].strip() if ":" in line else line

    return {
        "sentiment": sentiment,
        "explanation": explanation,
        "raw_response": response
    }

