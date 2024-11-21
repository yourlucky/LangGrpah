from typing import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage
import json

# Load API_key from .env file
from config_loader import ConfigLoader
config = ConfigLoader()

# select model GPT or Claude
def select_model():
    choice_map = {
        "1": ("Claude", ChatAnthropic(model="claude-3-5-sonnet-20240620")),
        "2": ("GPT-3", ChatOpenAI(model_name="gpt-3.5-turbo-1106"))
    }

    while (choice := input("Enter 1 for Claude or 2 for GPT-3: ").strip()) not in choice_map:
        print("Invalid choice. Please enter 1 or 2.")

    model_name, llm_instance = choice_map[choice]
    print(f"You selected {model_name}.")
    return llm_instance
    

# From QuickStart https://langchain-ai.github.io/langgraph/tutorials/introduction/#setup
########################################################################
class State(TypedDict):
    messages: Annotated[list, add_messages]    

graph_builder = StateGraph(State)

#Add New function; Tools
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = select_model()
llm_with_tools = llm.bind_tools(tools)

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

def chatbot(state: State):
    #llm.invoke에서 llm_with_tools.invoke로 변경
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools":"tools", END:END}
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
########################################################################

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

def main():
    print("Type 'quit', 'exit', or 'q' to exit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        try:
            stream_graph_updates(user_input)
        except Exception as e:
            print(f"Error occurred: {e}")
            break

# Run chatbot
if __name__ == "__main__":
    main()
    #Draw the graph
    # from IPython.display import Image, display
    # try:
    #     img = Image(graph.get_graph().draw_mermaid_png())
    #     print(img.data)
    #     with open("output.png", "wb") as f:
    #         f.write(graph.get_graph().draw_mermaid_png())
    # except Exception:
    #     # This requires some extra dependencies and is optional
    #     pass