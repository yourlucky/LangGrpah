# Load API_key from .env file
from config_loader import ConfigLoader

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

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
    
llm = select_model()


# From QuickStart https://langchain-ai.github.io/langgraph/tutorials/introduction/#setup
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
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