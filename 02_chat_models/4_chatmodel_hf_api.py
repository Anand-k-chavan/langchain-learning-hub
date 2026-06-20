from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

# Initialize the HuggingFaceEndpoint with your API key
hf_endpoint = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)

# Create a ChatHuggingFace instance using the endpoint
huggingface_chat_model = ChatHuggingFace(llm=hf_endpoint)

chat_result = huggingface_chat_model.invoke("What is the capital of India?")

print(chat_result)   # gives a more conversational response with metadata like tokens used, response time, etc.

print(chat_result.content)  # gives just the content of the response without metadata
