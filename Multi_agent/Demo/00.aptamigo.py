from typing import Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver

from node import CustomNode

# Load API_key from .env file
from config_loader import ConfigLoader
config = ConfigLoader()

# The agent state is the input to each node in the graph
class AgentState(MessagesState):
    # The 'next' field indicates where to route to next
    next: str


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
node = CustomNode(llm)
memory = MemorySaver()

#members = ["BudgetRecorder", "RoomRecorder"]
members=node.get_node_list()

# and decides when the work is completed
options = members+["FINISH"]

builder = StateGraph(AgentState)

builder.add_node("RealEstateAgent", node.RealEstateAgent_node)
builder.add_node("BudgetRecorder", node.budget_node)
builder.add_node("RoomRecorder", node.room_node)
builder.add_node("RelationshipBuilder", node.RelationshipBuilder_node)

builder.add_edge(START, "RealEstateAgent")
builder.add_conditional_edges("RealEstateAgent", lambda state: state["next"])
builder.add_edge("BudgetRecorder","RelationshipBuilder")
builder.add_edge("RoomRecorder","RelationshipBuilder")
builder.add_edge("RelationshipBuilder",END)

#graph = builder.compile()
graph = builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    for event in graph.stream({"messages": [("user", user_input)]},config, stream_mode="values"):
        #detail print
        #print(event)
        last_event = event
    if last_event and "messages" in last_event:
        last_message = last_event["messages"][-1]
        print(f"AptAmigo: {last_message.content}")
    
        

def main():
    print("Welcome to the AptAmigo Chatbot Demo!")
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

    #from IPython.display import Image
    #img = Image(graph.get_graph().draw_mermaid_png())
    #with open("output_pilot.png", "wb") as f:
    #    f.write(graph.get_graph().draw_mermaid_png())


# 
#  A house with 1 to 2 rooms would be ideal for me.
#  I’m looking for a house with exactly 3 rooms.
#  I’d like a home with 4 or more bedrooms.
#  I need a minimum of 2 rooms, but no more than 3.
#  A place with 2 to 5 rooms works best for my needs.

# My monthly budget is between $2,000 and $3,500.
# I don’t want to exceed $5,000 per month for rent.
# I’m comfortable spending $3,000 to $4,000 per month.
# My ideal budget range is $2,500 to $4,000.
# I’d like to stay within a budget of $3,500 per month or less.

# I’d like to move in by the first week of next month.
# I need to move in by the end of the month.