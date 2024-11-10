import * as dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { HumanMessage } from '@langchain/core/messages';

dotenv.config();

const llm = new ChatOpenAI({
  model: 'gpt-4o-mini',
  temperature: 0
});

const calculatorSchema = z.object({
  operation: z.enum(['add', 'subtract', 'multiply', 'divide']),
  number1: z.number().describe('The first number to operate on'),
  number2: z.number().describe('The second number to operate on')
});

const calculatorTool = tool(
  async ({ operation, number1, number2 }) => {
    console.log(operation, number1, number2);
    if (operation === 'add') {
      return `${number1 + number2}`;
    } else if (operation === 'subtract') {
      return `${number1 - number2}`;
    } else if (operation === 'multiply') {
      return `${number1 * number2}`;
    } else if (operation === 'divide') {
      return `${number1 / number2}`;
    } else {
      throw new Error('Invalid operation.');
    }
  },
  {
    name: 'calculator',
    description: 'Can perform mathematical operations',
    schema: calculatorSchema
  }
);


const llmWithTools = llm.bindTools([calculatorTool]);

const messages = [new HumanMessage("What is 3 * 12? Also, what is 11 + 49?")];

const aiMessage = await llmWithTools.invoke(messages);

messages.push(aiMessage);

const toolsByName = {
  calculator: calculatorTool
};

for (const toolCall of aiMessage.tool_calls) {
  const selectedTool = toolsByName[toolCall.name];
  const toolMessage = await selectedTool.invoke(toolCall);
  messages.push(toolMessage);
}

console.log(await llmWithTools.invoke(messages).then(res => res.content));