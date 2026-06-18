from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

m = InMemorySaver()


# Defining 5 different tools
@tool
def get_user_id(name: str):
    """Get the user ID for a given name."""
    return "123"


@tool
def get_user_address(user_id: str):
    """Get the address for a given user ID."""
    return "New York"


@tool
def get_weather(location: str):
    """Get the current weather for a location."""
    return "Sunny"


@tool
def get_shipping_cost(location: str):
    """Get the shipping cost to a location."""
    return "$10"


@tool
def check_inventory(item: str):
    """Check if an item is in stock."""
    return "In stock"


llm = ChatOllama(model="llama3.1", temperature=0)

agent = create_agent(
    model=llm,
    tools=[
        get_user_id,
        get_user_address,
        get_weather,
        get_shipping_cost,
        check_inventory,
    ],
    checkpointer=m,
    interrupt_before=["tools"], # <--- This is the modern LangGraph way to pause!
)

config = {"configurable": {"thread_id": "sam_laptop_thread"}}

print("=" * 50)
print("EXECUTING PURE AGENT (NO MEMORY, NO MIDDLEWARE)")
print("=" * 50)

prompt = """
I need to buy a 'laptop' for Sam. 
Please find Sam's user ID, then find his address, check the weather there, 
check the shipping cost to his address, and check if a 'laptop' is in stock.
Tell me all the details at the end.
"""

messages = [{"role": "user", "content": prompt}]


for event in agent.stream({"messages": messages}, config=config):
    for node_name, node_data in event.items():
        print(f" -> Node Executed: '{node_name}'")

# Check if the agent paused
state = agent.get_state(config)
if state.next and "tools" in state.next:
    print("\n   [INTERRUPT TRIGGERED] Graph paused right before executing tools!")
    
    # In a real app, you would ask the user for approval here.
    # We will simulate the user typing 'yes' and resuming the graph.
    input("\nPress Enter to approve the tool calls and resume the graph...")
    
    # To resume, we pass `None` to the stream with the exact same config
    for event in agent.stream(None, config=config):
        for node_name, node_data in event.items():
            print(f" -> Node Executed: '{node_name}'")
            
            if node_name == "tools":
                for msg in node_data.get("messages", []):
                    print(f"      [TOOL CALLED]: {msg.name} -> Result: {msg.content}")

print("\nFinal output can be found in the last 'model' node execution!")
