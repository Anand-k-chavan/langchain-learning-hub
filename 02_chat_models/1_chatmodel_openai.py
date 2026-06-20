from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI Chat Model
chat_model = ChatOpenAI(model="gpt-3.5-instruct", temperature=0.7, max_completion_tokens=150)

chat_result = chat_model.invoke("What is the capital of India?")

print(chat_result)   # gives a more conversational response with metadata like tokens used, response time, etc.

print(chat_result.content)  # gives just the content of the response without metadata   

