# model_response.py
from langchain_ollama import ChatOllama

def get_model_response(prompt: str) -> str:
    """
    Invokes the ChatOllama model with the given prompt and returns the response.
    """
    model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")
    response = model.invoke(prompt)
    return response
