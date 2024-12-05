from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState, START, END

from typing import List, Optional, Literal
from typing_extensions import TypedDict
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, trim_messages

class AgentState(MessagesState):
    next: str

def make_supervisor_node(llm: BaseChatModel, members: list[str], ) -> str:
    options = members
    
    system_prompt = (
        'You are an AptAmigo representative. Your role is solely to relay client information to the appropriate specialist with precision and without commentary.'
        'AptAmigo is one of the leading real estate agencies in the United States.'
        ' - For general inquiries or unrelated information, relay it to the "general_info_team".'
        ' - For any mention of tour dates or scheduling preferences, relay it to the "TourDateCoordinator".'
        ' Ensure that your responses are silent; you do not interact directly with the client beyond passing their information to the relevant specialist.'
        'Ensure that your responses are silent; you do not interact directly with the client beyond passing their information to the relevant specialist.'
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: MessagesState) -> MessagesState:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        next_ = response["next"]
        if next_ == "FINISH":
            next_ = END

        return {"next": next_}

    return supervisor_node



class CustomNode_1:
    def __init__(self, llm):
        self.llm = llm
        self.node_list =["BudgetRecorder", "RoomRecorder","RelationshipBuilder_1"]
        
        self.RealEstateAgent = (
            'You are an AptAmigo representative. '
            'Your role is to relay client information to the appropriate specialist with silence. '
            '- When the client shares budget information, relay it to the "BudgetRecorder". '
            '- When clients mention their preferred number of rooms, relay it to the "RoomRecorder". '
            '- For any other information or general conversation, relay it to the "RelationshipBuilder_1", who specializes in building rapport with the client. ' 
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
                "- Only output the JSON object in the exact format specified above. Do not include any explanations, comments, or additional text.\n"
                "- Extract the maximum budget as an integer from the input.\n"
                "- Use the exact budget range or highest value mentioned in the input.\n"
                "- Do not add any additional interpretations or assumptions.\n"
                "- Ensure the JSON object is valid and well-formed.\n\n"
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
                "Important:\n"
                "- Output **only** the JSON object. Do not include any other text.\n"
                "- Failure to strictly follow this rule will be considered incorrect."
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
                "- Only output the JSON object in the exact format specified above. Do not include any explanations, comments, or additional text.\n"
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
                "Important:\n"
                "- Output **only** the JSON object. Do not include any other text.\n"
                "- Failure to strictly follow this rule will be considered incorrect."
        )

        self.RelationshipBuilder = create_react_agent(
            llm,
            tools=[],
            state_modifier=
            "You are a Relationship Builder. Your role is to engage in brief, friendly conversation with the client, "
            "focusing on listening attentively and responding concisely in no more than two sentences. "
            "Acknowledge and empathize with the client's stories, interests, or concerns to make them feel valued. "
            "Additionally, as the conversation flows naturally, kindly and carefully ask the client about their preferred number of apartment rooms, monthly budget, or desired apartment tour date, one at a time. "
            "Approach these questions with sensitivity, ensuring the client feels comfortable and appreciated while you gather this information. "
            "Your goal is to create a welcoming and enjoyable experience while making the client feel heard and appreciated."
        )
    
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


    def Budget_node(self, state: AgentState) -> AgentState:
        result = self.BudgetRecorder.invoke(state)
        print("@@@budget_node@@@")
        print(result['messages'][-1].content)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="BudgetRecorder")
            ]
        }

    def Room_node(self, state: AgentState) -> AgentState:
        result = self.RoomRecorder.invoke(state)
        print("###room_node###")
        print(result['messages'][-1].content)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="RoomRecorder")
            ]
        }
    


class CustomNode_2:
    def __init__(self, llm):
        self.llm = llm
        self.node_list =["TourDateRecorder", "RelationshipBuilder_2"]

        self.TourDateCoordinator = (
                "You are an AptAmigo Tour Date Coordinator representative."
                "Your role is to derive high-quality information from the client regarding their preferred tour dates and times. "
                "Your primary task is to collect one or two specific dates that the client prefers for their apartment tour, along with the specific time slots. "
                "Please note that an apartment tour typically takes about 90 minutes, so it is recommended to schedule at least 2-3 hours between tours to avoid conflicts. "
                "Once all necessary details are collected, relay the information to the 'TourDateRecorder'. "
                "The 'TourDateRecorder' will record the dates and times in the following format: e.g., {\"tourDates\": [[\"2024-12-02T14:00:00\", \"2024-12-02T18:00:00\"]]}. "
                "If the client initially provides sufficient details that can be directly recorded by the 'TourDateRecorder', do not ask additional questions. Instead, immediately relay the information to the 'TourDateRecorder'. "
                "Remember to keep your responses concise, always using two sentences or fewer, and maintain a cheerful and respectful tone to ensure a positive client experience. "
                "For any other information or general conversation, relay it to the 'RelationshipBuilder_2', who specializes in building rapport with the client. "
                )
        
        self.TourDateRecorder = create_react_agent(
            llm,
            tools=[],
            state_modifier=
                "You are a Tour Date Recorder. Your task is to extract the desired tour dates mentioned by the user "
                "and provide the output in JSON format as follows:\n\n"
                "today date is " + datetime.now().strftime('%Y-%m-%d (%A)')+"\n\n"
                "Output Format:\n"
                "{\n"
                "  \"tourDates\": [\n"
                "    [\"start_datetime1\", \"end_datetime1\"],\n"
                "    [\"start_datetime2\", \"end_datetime2\"]\n"
                "  ]\n"
                "}\n\n"
                "Rules:\n"
                "- Only output the JSON object in the exact format specified above. Do not include any explanations, comments, or additional text.\n"
                "- Extract the dates and times from the user's input and convert them into ISO 8601 format (e.g., \"YYYY-MM-DDTHH:MM:SS\").\n"
                "- Each date should include both a start and an end time as a pair in a subarray (e.g., [\"2024-12-02T08:00:00\", \"2024-12-02T17:00:00\"]).\n"
                "- If the user provides a range of dates and times, split them into individual date-time pairs.\n"
                "- Ensure the output is a valid JSON object.\n"
                "- Do not include any additional fields or assumptions beyond the user's input.\n\n"
                "Examples:\n"
                "Input: I would like to tour apartments on December 2nd from 8 AM to 5 PM and December 3rd from 1 PM to 5 PM.\n"
                "Output:\n"
                "{\n"
                "  \"tourDates\": [\n"
                "    [\"2024-12-02T08:00:00\", \"2024-12-02T17:00:00\"],\n"
                "    [\"2024-12-03T13:00:00\", \"2024-12-03T17:00:00\"]\n"
                "  ]\n"
                "}\n\n"
                "Input: Can I schedule a tour on July 25th from 10 AM to 3 PM?\n"
                "Output:\n"
                "{\n"
                "  \"tourDates\": [\n"
                "    [\"2024-07-25T10:00:00\", \"2024-07-25T15:00:00\"]\n"
                "  ]\n"
                "}\n\n"
                "Input: I am free for a tour on January 15th in the afternoon from 13 to 16.\n"
                "Output:\n"
                "{\n"
                "  \"tourDates\": [\n"
                "    [\"2024-01-15T13:00:00\", \"2024-01-15T16:00:00\"]\n"
                "  ]\n"
                "}\n\n"
                "Important:\n"
                "- Output **only** the JSON object. Do not include any other text.\n"
                "- Failure to strictly follow this rule will be considered incorrect."
        )

        self.RelationshipBuilder = create_react_agent(
            llm,
            tools=[],
            state_modifier=
            "You are a Relationship Builder. Your role is to engage in brief, friendly conversation with the client, "
            "focusing on listening attentively and responding concisely in no more than two sentences. "
            "Acknowledge and empathize with the client's stories, interests, or concerns to make them feel valued. "
            "Additionally, as the conversation flows naturally, kindly and carefully ask the client about their preferred number of apartment rooms, monthly budget, or desired apartment tour date, one at a time. "
            "Approach these questions with sensitivity, ensuring the client feels comfortable and appreciated while you gather this information. "
            "Your goal is to create a welcoming and enjoyable experience while making the client feel heard and appreciated."
        )

    def get_router(self): 
        return TypedDict(
            "Router",
            {"next": Literal[tuple(self.node_list)]}, )


    def TourDateCoordinator_node(self,state: AgentState) -> AgentState:
        messages=[
            {"role": "system", "content": self.TourDateCoordinator},
        ] + state["messages"]
        response=self.llm.with_structured_output(self.get_router()).invoke(messages)
        next_ = response["next"]

        return {"next": next_}

    def TourDate_node(self, state: AgentState) -> AgentState:
        result = self.TourDateRecorder.invoke(state)
        print("*********tour_node***********")
        print(result['messages'][-1].content)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="TourDateRecorder")
            ]
        }
    
    def RelationshipBuilder_node(self,state: AgentState) -> AgentState:
        result = self.RelationshipBuilder.invoke(state)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="RelationshipBuilder")
            ]
        }

        

        
    

