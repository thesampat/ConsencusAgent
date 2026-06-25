from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


def get_llm(model: str, temp: float) -> ChatOllama:
    llm = ChatOllama(
        model="ministral-3:8b",
        temperature=0,
        num_predict=32768,
        num_ctx=8192,
    )
    #  num_ctx=32768
    return llm


def get_google_llm(
    model: str = "gemini-2.5-flash", temp: float = 0
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, temperature=temp)


def create_custom_agent(
    model, tools: list, system_prompt, check_pointer=None, **kwargs
):
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=check_pointer,
        system_prompt=system_prompt,
        **kwargs,
    )

    return agent
