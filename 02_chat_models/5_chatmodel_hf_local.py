from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

# Initialize the HuggingFacePipeline with the model and task
huggingface_pipeline = HuggingFacePipeline(
    model="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=100,
)

# Create a ChatHuggingFace instance using the pipeline
huggingface_chat_model = ChatHuggingFace(llm=huggingface_pipeline)

chat_result = huggingface_chat_model.invoke("What is the capital of India?")

print(chat_result)   # gives a more conversational response with metadata like tokens used, response time, etc.

print(chat_result.content)  # gives just the content of the response without metadata