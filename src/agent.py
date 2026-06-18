from langchain.agents.middleware.types import AgentMiddleware
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from Utilities.llm_model import get_llm, create_custom_agent
from dotenv import load_dotenv
from data import PROFILE_DATA, PREFERENCE_DATA
from langchain_core.runnables import RunnableConfig


load_dotenv()


m = InMemorySaver()
llm = get_llm("ollama3.1", 0)


@tool
def get_profile_data(user: str) -> str:
    """Get the profile details for the given user"""
    return PROFILE_DATA


@tool
def get_preferance_data(user: str):
    """Get the preferance data for the given user"""
    return PREFERENCE_DATA


preference_sub_agent = create_custom_agent(
    llm,
    [get_preferance_data],
    system_prompt="""You are a preference agent. Find and return the user's preferences.""",
    check_pointer=m,
)


profile_sub_agent = create_custom_agent(
    llm,
    [get_profile_data],
    system_prompt="""You are a profile agent. Find and return the user's profile.""",
    check_pointer=m,
)


# Sub Agent 1
@tool
def PreferanceAgent(question: str, config: RunnableConfig) -> str:
    """use this tool to query the preferance data of given user"""
    main_thread_id = config["configurable"].get("thread_id", "default_thread")
    res = preference_sub_agent.invoke(
        {"messages": [{"role": "user", "content": f"{question}"}]},
        config={"configurable": {"thread_id": f"{main_thread_id}_preference"}},
    )
    return res["messages"][-1].content


@tool
def ProfileAgent(question: str, config: RunnableConfig) -> str:
    """use this tool to query the profile data of given user"""
    main_thread_id = config["configurable"].get("thread_id", "default_thread")
    res = profile_sub_agent.invoke(
        {"messages": [{"role": "user", "content": f"{question}"}]},
        config={"configurable": {"thread_id": f"{main_thread_id}_profile"}},
    )
    return res["messages"][-1].content


agent = create_custom_agent(
    llm,
    [PreferanceAgent, ProfileAgent],
    system_prompt=(
        "If the answer to the user's question is already present in the chat history, do not call any tools. Answer directly using the historical information"
        "You are a super agent with access to sub-agents. Use your tools to answer user queries. "
        "IMPORTANT: Both 'PreferanceAgent' and 'ProfileAgent' tools ONLY accept a single parameter "
        "named 'question'. Do not pass 'user_id', 'user', or any other parameters to these tools. "
        "Only pass the 'question' parameter."
    ),
    check_pointer=m,
)


while True:
    print("---------ReInitializing Agent-------")
    i = input("ask question: ")
    if i == "exit":
        break

    mtream = agent.stream_events(
        {"messages": [{"role": "user", "content": i}]},
        config={"configurable": {"thread_id": "some1212_id_2"}},
        version="v3",
    )

    for message in mtream.messages:
        for delta in message.text:
            print(delta, end="", flush=True)
