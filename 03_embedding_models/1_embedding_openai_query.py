from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI Embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

# Generate embeddings for a sample text
embedding_result = embeddings_model.invoke("What is the capital of India?")

print(embedding_result)