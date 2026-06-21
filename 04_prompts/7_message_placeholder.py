from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat template with message placeholders
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}?'),
])

chat_history = []

# load chat history from a file
with open('04_prompts/chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

chat_template.invoke({
    'chat_history': chat_history, 
    'query': 'What is the status of my order.?'
    })