from langchain_openai import ChatOpenAI

def get_llm(model_name="gpt-4o", temperature=0.5):
    """OpenAI LLM 모델을 반환"""
    return ChatOpenAI(model_name=model_name, temperature=temperature)
