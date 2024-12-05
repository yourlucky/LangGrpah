from typing import Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver

from node import CustomNode_1, CustomNode_2, AgentState, make_supervisor_node

# Load API_key from .env file
from config_loader import ConfigLoader
config = ConfigLoader()


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
memory = MemorySaver()

node_1 = CustomNode_1(llm)
general_info = StateGraph(AgentState)

general_info.add_node("RealEstateAgent", node_1.RealEstateAgent_node)
general_info.add_node("BudgetRecorder", node_1.Budget_node)
general_info.add_node("RoomRecorder", node_1.Room_node)
general_info.add_node("RelationshipBuilder_1", node_1.RelationshipBuilder_node)

general_info.add_edge(START, "RealEstateAgent")
general_info.add_conditional_edges("RealEstateAgent", lambda state: state["next"])
general_info.add_edge("BudgetRecorder","RelationshipBuilder_1")
general_info.add_edge("RoomRecorder","RelationshipBuilder_1")
general_info.add_edge("RelationshipBuilder_1",END)

general_graph = general_info.compile(checkpointer=memory)

node_2 = CustomNode_2(llm)
tourdate_info = StateGraph(AgentState)

tourdate_info.add_node("TourDateCoordinator",node_2.TourDateCoordinator_node)
tourdate_info.add_node("TourDateRecorder", node_2.TourDate_node)
tourdate_info.add_node("RelationshipBuilder_2", node_2.RelationshipBuilder_node)

tourdate_info.add_edge(START, "TourDateCoordinator")
tourdate_info.add_conditional_edges("TourDateCoordinator", lambda state: state["next"])
tourdate_info.add_edge("RelationshipBuilder_2",END)

tourdate_graph = tourdate_info.compile(checkpointer=memory)

team_supervisor = make_supervisor_node(llm,["general_info_team","tourdate_info_team"])


def call_general_info(state: AgentState) -> AgentState:
    response = general_graph.invoke({"messages": state["messages"][-1]})
    return {
        "messages": [
            HumanMessage(content=response["messages"][-1].content, name="general_info_team")
        ]
    }

def call_tourdate_info(state: AgentState) -> AgentState:
    response = tourdate_graph.invoke({"messages": state["messages"][-1]})
    return {
        "messages": [
            HumanMessage(content=response["messages"][-1].content, name="tourdate_info_team")
        ]
    }

super_builder = StateGraph(AgentState)
super_builder.add_node("supervisor", team_supervisor)
super_builder.add_node("general_info_team", call_general_info)
super_builder.add_node("tourdate_info_team", call_tourdate_info)

super_builder.add_edge(START, "supervisor")
super_builder.add_conditional_edges("supervisor", lambda state: state["next"])
super_graph = super_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    #for s in general_graph.stream({"messages": [("user", user_input)]},config, stream_mode="values"):
    #for s in super_graph.stream({"messages": [("user", user_input)]},config, stream_mode="values"):
    #    print(s)
    
    for event in super_graph.stream({"messages": [("user", user_input)]},config, stream_mode="values"):
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

    # from IPython.display import Image
    # img = Image(super_graph.get_graph().draw_mermaid_png())
    # with open("output_pilot_general.png", "wb") as f:
    #    f.write(super_graph.get_graph().draw_mermaid_png())

