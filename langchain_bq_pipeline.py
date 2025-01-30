from typing import List, Union, Generator, Iterator
import os
from pydantic import BaseModel

# from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent
from utils.pipelines.main import get_last_assistant_message

class Pipeline:
    def __init__(self):
        self.name = "Database RAG Pipeline"
        self.engine = None

    def init_db_connection(self):
        # Initialize the BigQuery connection
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "atlas-poc-docai-sa.json")
        self.engine = SQLDatabase.from_uri("bigquery://mfec-dis-my/mf_atlas_poc_structure_raw_dataset")
        return self.engine

    async def on_startup(self):
        # This function is called when the server is started.
        self.init_db_connection()

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(self, user_message: str) -> Union[str, Generator, Iterator]:
        # Check if the database connection is initialized
        if self.engine is None:
            raise ValueError("Database connection is not initialized. Call init_db_connection() first.")

        # Create a SQLDatabase instance for LangChain
        sql_database = self.engine

        # Define the custom prompt for text-to-SQL
        SQL_PREFIX = """System: You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct bigquery query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        For the name 'XXX', it is a security name (stock name), not a company name.
        
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        
        To start you should ALWAYS look at the tables in the database to see what you can query.
        Do NOT skip this step.
        Then you should query the schema of the most relevant tables."""
        
        system_message = SystemMessage(content=SQL_PREFIX)

        # Initialize the LLM
        llm = ChatVertexAI(model="gemini-1.5-flash-002", temperature=0)

        # Set up the SQLDatabaseToolkit
        toolkit = SQLDatabaseToolkit(db=sql_database, llm=llm)
        tools = toolkit.get_tools()

        agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

        # Use the agent executor to process the input question
        response = agent_executor.invoke({"messages": [HumanMessage(content=user_message)]})

        # Yield each chunk from the response
        # for chunk in response:
            # yield chunk.content
        return response