import * as dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { z } from 'zod';
import { RunnableLambda } from "@langchain/core/runnables";
import { StringOutputParser } from '@langchain/core/output_parsers';

dotenv.config();

const model = new ChatOpenAI({
  model: 'gpt-4o-mini'
});

const promptTemplate = ChatPromptTemplate.fromMessages([
  ['system', 'Dont use punctuation and dont make gouda jokes'],
  ['human', 'Tell me a dark joke about {text}']
])

const joke = z.object({
  setup: z.string().describe('The setup of the joke'),
  punchline: z.string().describe('The body of the joke'),
  rating: z.number().describe('a rating of how good the joke is on a scale of 1-10')
});

const structuredLlm = model.withStructuredOutput(joke, { name: 'joke' });

const analysisPrompt = ChatPromptTemplate.fromTemplate(
  "is this a funny joke? {joke}"
);

const parser = new StringOutputParser();

const pipe = promptTemplate.pipe(structuredLlm);

const composedChain = new RunnableLambda({
  func: async (input) => {
    const result = await pipe.invoke(input);
    console.log(result);
    return { joke: result };
  }
})
.pipe(analysisPrompt)
.pipe(model)
.pipe(parser);

// console.log(await pipe.invoke({ text: 'cheese' }));
console.log(await composedChain.invoke({ text: 'cheese' }));
