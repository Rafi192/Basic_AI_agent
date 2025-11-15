from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import os

load_dotenv()

# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("anthropic_api_key")  # use your OpenRouter API key
# )

input_query = input("Enter your query: ")
# response = client.chat.completions.create(
#     model="anthropic/claude-3.5-sonnet",
#     messages=[
#         {"role": "user", "content": input_query}
#     ]
# )

# print(response.choices[0].message["content"])

genai.configure(api_key=os.getenv("gemini_api_key"))

model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content(
    input_query
)

print(response.text)