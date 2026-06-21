from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatOpenAI()

st.header("Research Tool")

paper_input = st.selectbox("Select a research paper:", [
    "Select...", "Attention Is All You Need", 
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 
    "GPT-3: Language Models are Few-Shot Learners", 
    "Diffusion Models Beat GANs on Image Synthesis", 
    "AlphaFold: Using AI for scientific discovery"
    ])

style_input = st.selectbox("Select Explaination Style", [
    "Beginner", "Intermediate", "Advanced"
    ])

lenght_input = st.selectbox("Select Explaination Length", [
    "Short(1-2 Paragraphs)", "Medium(3-4 Paragraphs)", "Long(5+ Paragraphs)  "
    ])

template = load_prompt('template.json')


if st.button('Summarize'):

    chain = template | model

# fill the placeholders
    result = template.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "lenght_input": lenght_input
    })
    st.write(result.content)