import {ChatOpenAI} from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";

let agentModel: ChatOpenAI | ChatAnthropic | undefined;

const getModel = () => {
    if (!agentModel) {
        agentModel = new ChatOpenAI({model: "gpt-4o", temperature: 0});
        //agentModel = new ChatAnthropic({model: "claude-3-5-sonnet-20241022",temperature: 0});
    }
    return agentModel;
};

export default getModel;