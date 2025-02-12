import {BaseMessage} from "@langchain/core/messages";
import {Annotation} from "@langchain/langgraph";

const StateAnnotation = Annotation.Root({
    sentiment: Annotation<string>,
    messageResponse: Annotation<BaseMessage[]>({
        reducer: (previous: BaseMessage[], incoming: BaseMessage | BaseMessage[]) => {
            return Array.isArray(incoming)
              ? previous.concat(incoming)
              : previous.concat([incoming]);
          },
        default: () => [],
    }),
    next: Annotation<string>({
        reducer: (prev, curr) => curr ?? prev,
        default: () => "END",
      }),
});

export default StateAnnotation;
