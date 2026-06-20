from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI Embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Bangalore is the IT hub of India."
    "Paris is the capital of France."
]

# Generate embeddings for a sample text
embedding_result = embeddings_model.embed_documents(documents)

print(str(embedding_result))