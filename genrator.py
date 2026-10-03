# generator.py
import os
from google import genai
from google.genai import types

def generate_answer(user_query, retrieved_context_str, api_key):
    """Calls Gemini API with strict system instructions and context."""
    resolved_api_key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
    if not resolved_api_key:
        return "⚠️ Error: Please provide a Gemini API Key or set GEMINI_API_KEY in Streamlit Secrets."

    system_instruction = (
        "You are an enterprise AI assistant. Answer the user question based on the provided context. "
        "If the user asks whether a feature, tool, or option is supported and it is missing from the supported list in the context, answer 'No'. "
        "If the context is completely unrelated to the question, say 'I cannot find the answer in the provided documents.'"
    )
    prompt = f"Context:\n{retrieved_context_str}\n\nUser Question: {user_query}"

    try:
        client = genai.Client(api_key=resolved_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "⚠️ Gemini generated an empty response or it was filtered by safety settings."
    except Exception as e:
        return f"⚠️ Gemini API Error: {str(e)}"