from typing import Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver


# Load API_key from .env file
from config_loader import ConfigLoader
config = ConfigLoader()

# select model GPT or Claude
def select_model():
    choice_map = {
        "1": ("Claude", ChatAnthropic(model="claude-3-5-sonnet-20240620")),
        "2": ("GPT-4o", ChatOpenAI(model_name="gpt-4o-2024-08-06"))
    }

    while (choice := input("Enter 1 for Claude or 2 for GPT-4o: ").strip()) not in choice_map:
        print("Invalid choice. Please enter 1 or 2.")

    model_name, llm_instance = choice_map[choice]
    print(f"You selected {model_name}.")
    return llm_instance

memory = MemorySaver()

# The agent state is the input to each node in the graph
class AgentState(MessagesState):
    # The 'next' field indicates where to route to next
    next: str


members = ["BudgetRecorder", "RoomRecorder"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    'You are an AptAmigo representative. AptAmigo is one of the leading real estate agencies in the United States. '
    'Your role is to engage with clients, build rapport, and make the conversation enjoyable while gathering their budget and preferred number of rooms. '
    '- When the client shares budget information, relay it to the "Budget Recorder". '
    '- When clients mention their preferred number of rooms, relay it to the "Room Recorder". '
    'When finished, respond with FINISH.'
)


class Router(TypedDict):
    """Worker to route to next."""

    next: Literal[*options]


llm = select_model()


def RealEstateAgent_node(state: AgentState) -> AgentState:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    if next_ == "FINISH":
        next_ = END

    return {"next": next_}



BudgetRecorder = create_react_agent(
    llm,
    tools=[],
    state_modifier=(
        "You are a Budget Recorder. Your task is to extract the maximum budget from the user's input "
        "and provide the output in JSON format as follows:\n\n"
        "Output Format:\n"
        "{\n"
        "  \"Budget\": {\n"
        "    \"max\": maximum_budget\n"
        "  },\n"
        "  \"original_input\": \"[original_user_sentence]\"\n"
        "}\n\n"
        "Rules:\n"
        "- Extract the maximum budget as an integer from the input.\n"
        "- Use the exact budget range or highest value mentioned in the input.\n"
        "- Do not add any additional interpretations or assumptions.\n"
        "- Ensure the output is a valid JSON object.\n\n"
        "Examples:\n"
        "Input: I can afford between $2,000 and $3,000 per month.\n"
        "Output: {\n"
        "  \"Budget\": {\n"
        "    \"max\": 3000\n"
        "  },\n"
        "  \"original_input\": \"I can afford between $2,000 and $3,000 per month.\"\n"
        "}\n\n"
        "Input: My budget is around $1500.\n"
        "Output: {\n"
        "  \"Budget\": {\n"
        "    \"max\": 1500\n"
        "  },\n"
        "  \"original_input\": \"My budget is around $1500.\"\n"
        "}\n\n"
        "Input: I don't want to spend more than $4,500 per month.\n"
        "Output: {\n"
        "  \"Budget\": {\n"
        "    \"max\": 4500\n"
        "  },\n"
        "  \"original_input\": \"I don't want to spend more than $4,500 per month.\"\n"
        "}\n\n"
        "Input: My budget range is $2,500 to $3,500.\n"
        "Output: {\n"
        "  \"Budget\": {\n"
        "    \"max\": 3500\n"
        "  },\n"
        "  \"original_input\": \"My budget range is $2,500 to $3,500.\"\n"
        "}\n\n"
        "Ensure all outputs strictly follow this JSON format."
    )
)


def budget_node(state: AgentState) -> AgentState:
    result = BudgetRecorder.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="BudgetRecorder")
        ]
    }

RoomRecorder = create_react_agent(
    llm,
    tools=[],
    state_modifier=
        "You are a Room Recorder. Your task is to extract the minimum and maximum number "
        "of rooms mentioned in the user's input and format the output as follows:\n\n"
        "Output Format:\n"
        "{\n"
        "  \"Room\": {\n"
        "    \"minimum\": minimum_rooms,\n"
        "    \"maximum\": maximum_rooms\n"
        "  },\n"
        "  \"original_input\": \"[original_user_sentence]\"\n"
        "}\n\n"
        "Rules:\n"
        "- Extract integers from the input to determine \"minimum\" and \"maximum\".\n"
        "- If only one number is provided, use it for both \"minimum\" and \"maximum\".\n"
        "- Do not interpret beyond the provided data.\n\n"
        "Examples:\n"
        "Input: I want a house with 2 to 3 rooms.\n"
        "Output: {\n"
        "  \"Room\": {\n"
        "    \"minimum\": 2,\n"
        "    \"maximum\": 3\n"
        "  },\n"
        "  \"original_input\": \"I want a house with 2 to 3 rooms.\"\n"
        "}\n\n"
        "Input: I need at least 1 room.\n"
        "Output: {\n"
        "  \"Room\": {\n"
        "    \"minimum\": 1,\n"
        "    \"maximum\": 1\n"
        "  },\n"
        "  \"original_input\": \"I need at least 1 room.\"\n"
        "}\n\n"
        "Input: I would prefer a house with more than 4 rooms.\n"
        "Output: {\n"
        "  \"Room\": {\n"
        "    \"minimum\": 4,\n"
        "    \"maximum\": 4\n"
        "  },\n"
        "  \"original_input\": \"I would prefer a house with more than 4 rooms.\"\n"
        "}\n\n"
        "Input: I need a house with a minimum of 1 room and a maximum of 5 rooms.\n"
        "Output: {\n"
        "  \"Room\": {\n"
        "    \"minimum\": 1,\n"
        "    \"maximum\": 5\n"
        "  },\n"
        "  \"original_input\": \"I need a house with a minimum of 1 room and a maximum of 5 rooms.\"\n"
        "}\n\n"
        "Follow these rules and examples strictly for all inputs."
)

def room_node(state: AgentState) -> AgentState:
    result = RoomRecorder.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="RoomRecorder")
        ]
    }

builder = StateGraph(AgentState)
builder.add_edge(START, "RealEstateAgent")
builder.add_node("RealEstateAgent", RealEstateAgent_node)
builder.add_node("BudgetRecorder", budget_node)
builder.add_node("RoomRecorder", room_node)

builder.add_conditional_edges("RealEstateAgent", lambda state: state["next"])
# Finally, add entrypoint
builder.add_edge(START, "RealEstateAgent")
builder.add_edge('BudgetRecorder',END)
builder.add_edge('RoomRecorder',END)


graph = builder.compile(checkpointer=memory)
#graph = builder.compile()

def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    print("here")
    for event in graph.stream({"messages": [("user", user_input)]},config, stream_mode="values"):
        print(event)
        #for value in event.values():
        #    print("Assistant:", value["messages"][-1].content)

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
    #     with open("output_pilot.png", "wb") as f:
    #         f.write(graph.get_graph().draw_mermaid_png())
    # except Exception:
    #     # This requires some extra dependencies and is optional
    #     pass

# for s in graph.stream(
#     {"messages": [("user", "I can afford up to $2000 per month.")]}, #subgraphs=True
# ):
#     print(s)
#     print("----")