from langchain_openai import ChatOpenAI

def get_llm(model_name="gpt-4.1-mini", temperature=0.0):
    """OpenAI LLM 모델을 반환"""
    return ChatOpenAI(model_name=model_name, temperature=temperature)
