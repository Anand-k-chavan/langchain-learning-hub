from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}.',
    input_variables=['topic']
)

model = HuggingFaceEmbeddings(model="gemma3:1b")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'volleyball'})

print(result)