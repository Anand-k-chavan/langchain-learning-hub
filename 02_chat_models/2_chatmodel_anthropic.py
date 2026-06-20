from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# Initialize the Anthropic Chat Model
anthropic_chat_model = ChatAnthropic(model="claude-2", temperature=0.7)

anthropic_result = anthropic_chat_model.invoke("What is the capital of India?")

print(anthropic_result)   # gives a more conversational response with metadata like tokens used, response time, etc.

print(anthropic_result.content)  # gives just the content of the response without metadata