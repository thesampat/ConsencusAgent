from IPython.core import compilerop
from Utilities.llm_model import get_google_llm
from typing import Literal, Optional, Annotated
import operator
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from Utilities.llm_model import get_llm
import json


load_dotenv()

# model: ChatOllama = get_google_llm()
model: ChatOllama = get_llm(model="mistral-3:8b", temp=0)


class StructuredItem(BaseModel):
    movie: Optional[str] = None
    director: Optional[str] = None
    genre: Optional[str] = None

    track: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None


class ItemList(BaseModel):
    varities: list[StructuredItem]


@tool
def searchh_items():
    """search tool"""
    return [
        {"track": "Shape of You G7"},
        {"director": "Christopher Nolan X2"},
        {"artist": "Ed Sheeran G7"},
        {"movie": "Interstellar X2"},
        {"genre": "Sci-Fi X2"},
        {"album": "Divide G7"},
        {"artist": "Adele M5"},
        {"album": "25 M5"},
        {"track": "Hello M5"},
        {"genre": "Crime D9"},
        {"movie": "The Godfather D9"},
        {"director": "Francis Ford Coppola D9"},
        {"weather": "Sunny"},
        {"temperature": "30C"},
        {"random": "Ignore me"},
    ]


tools_by_name = {
    "searchh_items": searchh_items,
}


model_with_tools = model.bind_tools([searchh_items])
model_with_structure = model.with_structured_output(ItemList)


messages2 = [
    SystemMessage(
        content="You are a producer. You MUST call the tool to retrieve the list of items"
    ),
    HumanMessage(content="give me list"),
]


res = model_with_tools.invoke(messages2)

tool_data = []

print(res)
if res.tool_calls:
    messages2.append(res)
    for tool_call in res.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call)
        tool_data.extend(json.loads(tool_result.content))


formatting_prompt = [
    SystemMessage(
        content="""
        You are given the output of a tool.

Your task is to discover how the records are related and organize them into structured objects.

Rules:

1. Use ONLY the provided data.
2. Do NOT use external knowledge.
3. Do NOT invent, infer, or modify any value.
4. Discover the hidden grouping relationship that exists in the data.
5. Records belonging to the same logical group must be combined into exactly one object.
6. Every field in an object MUST originate from the same group.
7. Never copy or borrow a field from another group.
8. If a field does not exist for a group, return null instead of guessing.
9. Ignore records that cannot be associated with any group.
10. Remove any grouping markers or metadata from the returned values if they are only used for grouping.
11. Return ONLY the final structured output.
12. Do NOT explain your reasoning.
13. Do NOT include intermediate structures such as grouping_marker, items, fields, notes, markdown, or code fences.

Before producing the final answer, internally perform these steps:

- Identify every unique logical group.
- Collect every record belonging to that group.
- Verify each record belongs to only one group.
- Verify no field has been copied from another group.
- Create exactly one structured object for the completed group.
- Repeat until every group has been processed.

        """
    ),
    HumanMessage(content=json.dumps(tool_data, indent=2)),
]

print(tool_data)


final_output = model.invoke(formatting_prompt)

parser_prompt = [
    SystemMessage(
        content="""
        return the list, don't assume predict values, use only recived data from the tool
"""
    ),
    HumanMessage(content=final_output.content),
]


print(final_output.content)
structured_output = model_with_structure.invoke(parser_prompt)


# print("\n--- Final Structured Output ---")
print(structured_output)
