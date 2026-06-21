#template
from langchain_core.prompts import PromptTemplate


prompt_template = PromptTemplate(
    template="""
Please summerize the research paper titled '{paper_input}' with the following specifications:
Explaination Style: {style_input}
Explaination Length: {lenght_input}
1. Mathemantical Details:
-Include relevant mathematical equations and derivations if present in the paper.
-Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
-Use relatable analogies to explain complex concepts, making them easier to understand.
If certain information is not available in the paper, please mention that it is not available
 or Insufficient information is provided in the paper instead of guessing.
Ensure the summary is clear, concise, and accessible to 
readers with varying levels of expertise in the subject matter.
""",
input_variables=["paper_input", "style_input", "lenght_input"]
)

prompt_template.save('template.json')