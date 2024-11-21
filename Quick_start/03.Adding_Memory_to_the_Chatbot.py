from typing import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver


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
    
####Adding Memory to the Chatbot
memory = MemorySaver()

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

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
###*****add New*****
graph = graph_builder.compile(checkpointer=memory)

########################################################################


def stream_graph_updates_simple(user_input: str, user_memory: str):
    config = {"configurable": {"thread_id": user_memory}}
    for event in graph.stream({"messages": [("user", user_input)]},config, stream_mode="values"):
        last_value = list(event.values())[-1]
        if isinstance(last_value, list):
            last_message = last_value[-1]
            if isinstance(last_message, AIMessage):
                print(f"AI Message_{user_memory}: {last_message.content}") 

def main():
    user_input = input("User memory number ? : ")
    current_memory = user_input
    print(f"You are using memory {current_memory}")



    print("Type 'quit', 'exit', or 'q' to exit.")
    print("Type '1', '2', or '3' to change memory.")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            if user_input in ["1", "2"]:
                new_memory = int(user_input)
                if new_memory != current_memory:
                    print(f"Switching to memory {new_memory}")
                    current_memory = new_memory
                continue
        
            stream_graph_updates_simple(user_input, current_memory)

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