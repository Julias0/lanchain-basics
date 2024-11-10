import * as dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from "@langchain/core/prompts";

dotenv.config();

const model = new ChatOpenAI({
  model: 'gpt-4o-mini'
});
const parser = new StringOutputParser();

const systemTemplate = 'Translate the following to {language}lish in English';
const promptTemplate = ChatPromptTemplate.fromMessages([
  ['system', systemTemplate],
  ['human', '{text}']
]);

const pipe = promptTemplate.pipe(model).pipe(parser);

console.log(await pipe.invoke({
  language: 'Kannada',
  text: 'Hi! How are you?'
}));



