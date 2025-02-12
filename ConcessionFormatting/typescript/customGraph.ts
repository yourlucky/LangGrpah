import { MemorySaver, StateGraph } from "@langchain/langgraph";
import customNode from "./customNode";
import getModel from "./getModel";
import StateAnnotation from "./annotation";

const model = getModel();

export const customGraph = (memorySaver: MemorySaver) => {
  const workflow = new StateGraph(StateAnnotation)
      .addNode("customNode", (state) => customNode(state, memorySaver, model))
      .addEdge("__start__", "customNode")
      .addEdge("customNode", "__end__");

  return workflow.compile(memorySaver);
};