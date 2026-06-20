from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    'Delhi is the capital of India.',
    'Mumbai is the financial capital of India.',
    'Bangalore is the IT hub of India.',
    'Paris is the capital of France.'
    'Tokyo is the capital of Japan.'
]

# Generate embeddings for the documents
query = "What is the capital of France?"
query_embedding = embeddings_model.invoke(query)

document_embeddings = embeddings_model.embed_documents(documents)
query_embedding = embeddings_model.embed_query(query)

cosine_similarities = cosine_similarity([query_embedding], document_embeddings)[0]

index, score = sorted(list(enumerate(cosine_similarities)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])  # Output: "Paris is the capital of France."
print("Similarity Score:", score)  # Output: Similarity Score: 0.95 (example score)
