from typing import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

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


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = select_model()
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()


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