from langchain_ollama import ChatOllama
from langchain.agents import create_agent


def get_llm(model: str, temp: float):
    llm = ChatOllama(model="llama3.1", temperature=0)
    return llm


def create_custom_agent(model, tools: list, system_prompt, check_pointer=None):
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=check_pointer,
        system_prompt=system_prompt,
    )

    return agent
