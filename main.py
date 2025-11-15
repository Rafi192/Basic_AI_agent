from dotenv import load_dotenv
from pydantic import BaseModel
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()

# -----------------------------
# LLM: Gemini 2.5 Flash
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("gemini_api_key"), # Standard env var name
    temperature=0.2,
)

# -----------------------------
# Pydantic output model (optional - for structured output)
# -----------------------------
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# -----------------------------
# Example tool
# -----------------------------
@tool
def dummy_tool(q: str) -> str:
    """Simple tool that echoes the input query."""
    return f"Tool response: {q}"

tools = [dummy_tool]

# -----------------------------
# CREATE AGENT
# -----------------------------
# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI agent. Use the available tools to help answer questions."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# -----------------------------
# RUN AGENT
# -----------------------------
if __name__ == "__main__":
    input_query = input("Enter your query: ")
    
    try:
        response = agent_executor.invoke({"input": input_query})
        print("\n=== RESPONSE ===")
        print(response)
    except Exception as e:
        print(f"Error: {e}")