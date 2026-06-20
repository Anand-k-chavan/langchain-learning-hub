from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the Google GenAI Chat Model
google_chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

google_result = google_chat_model.invoke("What is the capital of India?")

print(google_result)   # gives a more conversational response with metadata like tokens used, response time, etc.

print(google_result.content)  # gives just the content of the response without metadata