from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from data import PROFILE_DATA, PREFERENCE_DATA

load_dotenv()


# 1. Define the Shared State
class AgentState(TypedDict):
    # add_messages appends new messages to the existing list
    messages: Annotated[list, add_messages]
    # Shared keys where worker nodes write their findings
    profile_info: str
    preference_info: str


# Initialize LLM
llm = ChatOllama(model="llama3.1", temperature=0)


# Helper function to extract user name from message history
def _extract_user(messages: list) -> str:
    for m in reversed(messages):
        if not m.content:
            continue
        content = m.content.lower()
        for u in PROFILE_DATA.keys():
            if u in content:
                return u
    return "sam"  # Default fallback


# 2. Worker Node: Profile Agent
def profile_node(state: AgentState):
    print("\n[Executing Node: Profile Agent]")
    user = _extract_user(state["messages"])
    data = PROFILE_DATA.get(user, "User profile not found")
    return {"profile_info": str(data)}


# 3. Worker Node: Preference Agent
def preference_node(state: AgentState):
    print("\n[Executing Node: Preference Agent]")
    user = _extract_user(state["messages"])
    data = PREFERENCE_DATA.get(user, "User preferences not found")
    return {"preference_info": str(data)}


# 4. Routing logic: Determines which node to visit next
def router(state: AgentState):
    last_message = state["messages"][-1].content.lower()

    needs_profile = any(
        w in last_message for w in ["live", "username", "place", "profile", "email"]
    )
    needs_pref = any(
        w in last_message
        for w in ["food", "eat", "drink", "hobby", "color", "preference"]
    )

    # Route dynamically based on what info the query needs vs what we have in state
    if needs_profile and not state.get("profile_info"):
        return "profile"
    if needs_pref and not state.get("preference_info"):
        return "preference"

    return "generator"


# 5. Generator Node: Produces the final answer using the compiled state
def generator_node(state: AgentState):
    print("\n[Executing Node: Generator]")
    profile = state.get("profile_info", "No profile data loaded.")
    prefs = state.get("preference_info", "No preference data loaded.")
    user_query = state["messages"][-1].content

    prompt = f"""You are a helpful assistant.
Using the collected context below, answer the user's question directly.

Context:
- Profile: {profile}
- Preferences: {prefs}

User Question: {user_query}"""

    res = llm.invoke(prompt)
    return {"messages": [res]}


# 6. Build the State Graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("profile", profile_node)
builder.add_node("preference", preference_node)
builder.add_node("generator", generator_node)

# Add conditional routing from START
builder.add_conditional_edges(
    START,
    router,
    {"profile": "profile", "preference": "preference", "generator": "generator"},
)

# Route back to router from worker nodes to see if more info is needed
builder.add_conditional_edges(
    "profile",
    router,
    {
        "profile": "profile",  # fallback loop
        "preference": "preference",  # go query preference if also needed
        "generator": "generator",  # compile final answer
    },
)

builder.add_conditional_edges(
    "preference",
    router,
    {"profile": "profile", "preference": "preference", "generator": "generator"},
)

# Generator completes the graph execution
builder.add_edge("generator", END)

# Compile graph with memory saver
memory = InMemorySaver()
agent = builder.compile(checkpointer=memory)

# 7. Interactive Run Loop
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "shared_state_demo_thread"}}
    print("=== Shared State Agent Initialized ===")

    while True:
        user_input = input("\nAsk question (or 'exit'): ")
        if user_input.lower() == "exit":
            break

        events = agent.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        )

        for event in events:
            # Print state updates at each step
            for node_name, state_update in event.items():
                if "messages" in state_update:
                    print(f"Response: {state_update['messages'][-1].content}")
