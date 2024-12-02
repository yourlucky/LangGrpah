from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END

from typing import Literal, Union
from typing_extensions import TypedDict

#이거 삭제하는거 실험
class AgentState(MessagesState):
    next: str

class CustomNode:
    def __init__(self, llm):
        self.llm = llm
        self.node_list =["BudgetRecorder", "RoomRecorder","RelationshipBuilder","TourDateRecorder"]
        
        self.RealEstateAgent = (
            'You are an AptAmigo representative. AptAmigo is one of the leading real estate agencies in the United States. '
            'Your role is to relay client information to the appropriate specialist while ensuring a friendly and enjoyable experience for the client. '
            '- When the client shares budget information, relay it to the "Budget Recorder". '
            '- When clients mention their preferred number of rooms, relay it to the "Room Recorder". '
            '- When clients mention their preferred tour dates, relay it to the "Tour TourDate Recorder". '
            '- For any other information or general conversation, relay it to the "Relationship Builder", who specializes in building rapport with the client. ' 
        )


        self.BudgetRecorder = create_react_agent(
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

        self.RoomRecorder = create_react_agent(
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
        self.RelationshipBuilder = create_react_agent(
            llm,
            tools=[],
            state_modifier=
            "You are a Relationship Builder. Your role is to engage in brief, friendly conversation with the client, "
            "focusing on listening attentively and responding concisely in no more than two sentences. "
            "Acknowledge and empathize with the client's stories, interests, or concerns to make them feel valued. "
            "Additionally, if the opportunity arises, kindly ask the client about their preferred number of apartment rooms, monthly budget, and desired apartment tour date to better assist them."
            "Your goal is to create a welcoming and enjoyable experience while making the client feel heard and appreciated."
        )

        self.TourDateRecorder = create_react_agent(
            llm,
            tools=[],
            state_modifier=(
                "You are a Tour Date Recorder. Your task is to extract the desired tour dates mentioned by the user "
                "and provide the output in JSON format as follows:\n\n"
                "Output Format:\n"
                "{\n"
                "  \"TourDates\": [\n"
                "    \"date1\",\n"
                "    \"date2\",\n"
                "    \"date3\"\n"
                "  ],\n"
                "  \"original_input\": \"[original_user_sentence]\"\n"
                "}\n\n"
                "Rules:\n"
                "- Extract the dates as strings in the format 'MM/DD' or 'YYYY/MM/DD' from the input.\n"
                "- If multiple dates are mentioned, add them to a list.\n"
                "- If the same date is mentioned multiple times, include it only once.\n"
                "- Ensure the output is a valid JSON object.\n"
                "- Retain the original input text in the output.\n\n"
                "Examples:\n"
                "Input: I would like to tour apartments on July 25th, 26th, and 27th.\n"
                "Output: {\n"
                "  \"TourDates\": [\"07/25\", \"07/26\", \"07/27\"],\n"
                "  \"original_input\": \"I would like to tour apartments on July 25th, 26th, and 27th.\"\n"
                "}\n\n"
                "Input: Can I schedule a tour for 2024/07/28?\n"
                "Output: {\n"
                "  \"TourDates\": [\"2024/07/28\"],\n"
                "  \"original_input\": \"Can I schedule a tour for 2024/07/28?\"\n"
                "}\n\n"
                "Input: I want to tour apartments on 08/01, 08/02, and 08/03.\n"
                "Output: {\n"
                "  \"TourDates\": [\"08/01\", \"08/02\", \"08/03\"],\n"
                "  \"original_input\": \"I want to tour apartments on 08/01, 08/02, and 08/03.\"\n"
                "}\n\n"
                "Ensure all outputs strictly follow this JSON format and avoid any assumptions beyond the user's input."
            )
    )
    
    def get_node_list(self):
        return ["BudgetRecorder", "RoomRecorder","RelationshipBuilder"]
    
    def get_router(self): 
        return TypedDict(
            "Router",
            {"next": Literal[tuple(self.node_list)]}, )
    
    def RelationshipBuilder_node(self,state: AgentState) -> AgentState:
        result = self.RelationshipBuilder.invoke(state)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="RelationshipBuilder")
            ]
        }
        
    def RealEstateAgent_node(self,state: AgentState) -> AgentState:
        messages = [
            {"role": "system", "content": self.RealEstateAgent},
        ] + state["messages"]
        response=self.llm.with_structured_output(self.get_router()).invoke(messages)
        next_ = response["next"]
        
        return {"next": next_}



    def budget_node(self, state: AgentState) -> AgentState:
        result = self.BudgetRecorder.invoke(state)
        print("@@@@@@@@@@@@@@@@")
        print(result['messages'][-1].content)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="BudgetRecorder")
            ]
        }

    def room_node(self, state: AgentState) -> AgentState:
        result = self.RoomRecorder.invoke(state)
        print("###################")
        print(result['messages'][-1].content)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="RoomRecorder")
            ]
        }
    def tour_node(self, state: AgentState) -> AgentState:
        result = self.BudgetRecorder.invoke(state)
        print("***************************")
        print(result['messages'][-1].content)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="TourDateRecorder")
            ]
        }
    
