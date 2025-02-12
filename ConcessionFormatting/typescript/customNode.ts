import {createReactAgent} from "@langchain/langgraph/prebuilt";
import { SystemMessage} from "@langchain/core/messages";
import getModel from "./getModel";
import StateAnnotation from "./annotation";

type GraphState = typeof StateAnnotation;

const customNode = async (state: GraphState, memorySaver: any, model: ReturnType<typeof getModel>
) => {
  const SYSTEM_TEMPLATE = `You are the best real estate rental agent at AptAmigo. 
            Your primary goal is to build rapport with clients and understand their preferences apartment tourdate. 
            Listen attentively and respond concisely, keeping your answers to no more than two sentences.`;
  
  const systemMessage = new SystemMessage(SYSTEM_TEMPLATE);

  // Remove any previous system messages matching the template from the state's messageResponse.
  const filteredMessages = Array.isArray(state.messageResponse)
    ? state.messageResponse.filter((msg) => msg.content !== SYSTEM_TEMPLATE)
    : [];

  // 최종 메시지 배열
  const messages = [systemMessage, ...filteredMessages];


  // Create the React Agent
  const bot = createReactAgent({
    llm: model,
    tools:[],
    checkpointSaver : memorySaver,
  })

  const response = await bot.invoke({ messages }, { configurable: { thread_id: "1" } });

  return {
    //messageResponse: response.messages[response.messages.length - 1],
    //messageResponse: response.messages.at(-1),
    messageResponse: response.messages,
  };
}

export default customNode;