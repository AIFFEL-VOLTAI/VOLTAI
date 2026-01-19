from langchain_openai import ChatOpenAI
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(model_name="gemini-2.5-pro"):
    """OpenAI LLM 모델을 반환"""
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.0)