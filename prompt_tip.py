#Dynamic Prompt
#We can receive the items to be collected from the database as a string and create a dynamic prompt.


#First example
def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

#Second example
from datetime import datetime
current_date_with_day = datetime.now().strftime('%Y-%m-%d (%A)')
print(current_date_with_day)

TourDateRecorder = create_react_agent(
            llm,
            tools=[],
            state_modifier=
                "You are a Tour Date Recorder. Your task is to extract the desired tour dates mentioned by the user "
                "and provide the output in JSON format as follows:\n\n"
                "today date is " + datetime.now().strftime('%Y-%m-%d (%A)')+"\n\n"
                "Output Format:\n"
                "{\n"
                )
