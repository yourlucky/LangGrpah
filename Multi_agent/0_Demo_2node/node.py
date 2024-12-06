from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END

from typing import Literal, Union
from typing_extensions import TypedDict
from datetime import datetime


class AgentState(MessagesState):
    next: str

class CustomNode:
    def __init__(self, llm):
        self.llm = llm
        self.node_list =["RelationshipBuilder","TourDateRecorder"]
        
        self.RealEstateAgent = (
            "You are an AptAmigo representative. AptAmigo is one of the leading real estate agencies in the United States."
            "Your role is to ensure all necessary client information is collected and relayed to the appropriate specialist."
            "- If the client only mentions a preferred tour date without specifying a time, do not relay the information directly to the 'TourDateRecorder'. "
            "  Instead, pass the inquiry to the 'RelationshipBuilder' to gather the specific time. "
            "- Once both the desired tour date and time are confirmed, relay the complete information to the 'TourDateRecorder'. "
            '- For any other information or general conversation, relay it to the "RelationshipBuilder", who specializes in building rapport with the client. '
            " Your goal is to streamline the process while ensuring the client feels heard and supported with silent."
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
            " - If the client shares their desired tour date but not the specific time, follow up with: "
            "'Could you let me know the specific time you’re available for the tour on [date]?' "
            "Additionally, as the conversation flows naturally, kindly and carefully ask the client about their desired apartment tour date. "
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

    
    def Tour_node(self, state: AgentState) -> AgentState:
        result = self.TourDateRecorder.invoke(state)
        print("*********tour_node***********")
        print(result['messages'][-1].content)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="TourDateRecorder")
            ]
        }
    
    
