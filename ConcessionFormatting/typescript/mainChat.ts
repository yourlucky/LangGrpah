import { MemorySaver } from "@langchain/langgraph";
import { BaseMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import { customGraph } from "./customGraph";

const getLastAIResponse = (state: { messageResponse: BaseMessage[] }): string => {
    const reversedMessages = [...state.messageResponse].reverse();
    const aiMessage = reversedMessages.find((message: BaseMessage) => message instanceof AIMessage);
    if (!aiMessage || !("content" in aiMessage) || !aiMessage.content) {
        throw new Error("No AI response found");
      }
      return aiMessage.content as string;
    };

// Main function to initialize and run the chatbot
const main = async () => {

    const memorySaver = new MemorySaver();
    const graph = customGraph(memorySaver);

    console.log("Welcome to the AptAmigo Chatbot Demo!");
    const readline = require("readline").createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    const promptUser = async () => {
        readline.question("User: ", async (userInput: string) => {
            if (["quit", "exit", "q"].includes(userInput.toLowerCase())) {
                console.log("Goodbye!");
                readline.close();
                return; 
            }
    
            try {
                const state = await graph.invoke({
                    messageResponse: [new HumanMessage(userInput)],
                });
                const res = await getLastAIResponse(state);
                console.log("AI: ", res);
            } catch (error) {
                if(error instanceof Error){
                    console.error(`Error occurred: ${error}`);
                    console.log(error.stack);
                }
                readline.close();
            } finally {
                if (readline.listenerCount("line") > 0) {
                    readline.removeListener("line", promptUser);
                }
                promptUser();
            }
        });
    };
    promptUser();
};

main();



    //import { config } from "dotenv";
//import { resolve } from "path";

// // env 캐쉬 삭제
// delete process.env.OPENAI_API_KEY;

// // `.env` 파일 로드
// const envPath = resolve(process.cwd(), ".env");
// config({ path: envPath });

// function getEnvPath(): string {
//     return `📂 .env file is located at: ${envPath}`;
// }

// function getOpenAIKey(): string {
//     const apiKey = process.env.OPENAI_API_KEY;
//     return apiKey ? `🔑 OPENAI_API_KEY: ${apiKey}` : "❌ OPENAI_API_KEY not found!";
// }

// console.log(getEnvPath());
// console.log(getOpenAIKey());
