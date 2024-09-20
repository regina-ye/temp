from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from get_embedding_function import get_embedding_function

#load the data
persist_directory = "chroma"
db = Chroma(persist_directory=persist_directory, embedding_function=get_embedding_function())

#retrieve relevant documents
query = input("Enter your question about event: ")
#exmaple questions:
# When is 50 Meter happening?
relevant_events = db.similarity_search(query)

context_text = "\n\n".join(f"Event {i+1}:\n{event}" for i, event in enumerate(relevant_events))

prompt = f"""
Answer the question based only on the following context, as briefly as possible:
{context_text}
---
Question: {query}
Answer:
"""

model = Ollama(model="mistral")
response = model.invoke(prompt)
print("Context: ")
print(context_text)
print("Response: ")
print(response)
