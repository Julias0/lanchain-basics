import * as dotenv from "dotenv";
import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";
import { ChatOpenAI } from "@langchain/openai";
import { createSqlQueryChain } from "langchain/chains/sql_db";
import { QuerySqlTool } from "langchain/tools/sql";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';

dotenv.config();

const dataSource = new DataSource({
  type: "sqlite",
  database: "./sqlite.db",
});

const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: dataSource,
});

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  verbose: true
});

const customPromptTemplate = new PromptTemplate({
  inputVariables: ["input", "table_info", "top_k", "dialect"],
  template: `
  Double check the user's {dialect} query for common mistakes, including:
  - Only return SQL Query not anything else like \`\`\`sql ... \`\`\`
  - Using NOT IN with NULL values
  - Using UNION when UNION ALL should have been used
  - Using BETWEEN for exclusive ranges
  - Data type mismatch in predicates\
  - Using the correct number of arguments for functions
  - Casting to the correct data type
  - Using the proper columns for joins
  
  If there are any of the above mistakes, rewrite the query.
  If there are no mistakes, just reproduce the original query with no further commentary.
  
  Output the final SQL query only.

  {table_info}

  Question: {input}
`
})

const writeQuery = await createSqlQueryChain({
  llm,
  db,
  dialect: "sqlite",
  prompt: customPromptTemplate
});

const executeQuery = new QuerySqlTool(db);

// const response = await writeQuery.pipe(executeQuery).invoke({
//   question: "What is the hints column about?",
// });

// console.log("db run result", response);

const answerPrompt = PromptTemplate.fromTemplate(
  `Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: `
);

const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());

// const finalChain = RunnableSequence.from([
//   RunnablePassthrough
//   .assign({
//     query: writeQuery
//   })
//   .assign({
//     result: (i) => executeQuery.invoke(i.query),
//   }),
//   answerChain
// ]);

// console.log(await finalChain.invoke({ question: "How many employees are there" }));


const finalChain = RunnableSequence.from([
  RunnablePassthrough.assign({ query: writeQuery }).assign({
    result: (i) => executeQuery.invoke(i.query),
  }),
  answerChain,
]);
console.log(await finalChain.invoke({ question: "What is the overall topic of the single table present" }));