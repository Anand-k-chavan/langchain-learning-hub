from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOllama(model="gemma3:1b")

model2 = ChatHuggingFace(model="gemma3:1b")

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}.',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text: {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the following two texts into a single coherent text: {text1} and {text2}',
    input_variables=['text1', 'text2']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'text1': prompt1 | model1 | parser,
    'text2': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'topic': 'Unemployement in India'})

print(result)

chain.get_graph().print_ascii()

