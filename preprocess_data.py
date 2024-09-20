#parts of code from Pixegami's repository here: https://github.com/pixegami/rag-tutorial-v2
 
import os
import re
import csv
import json
import urllib.request
from langchain.docstore.document import Document
from get_embedding_function import get_embedding_function
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"

# load events data from tamu in json format
with urllib.request.urlopen("https://calendar.tamu.edu/live/json/events/group") as url:
  data = json.load(url)

def strip_html(text):
  """ Remove HTML tags from a string """
  if not isinstance(text, str):
    return ''
  clean = re.compile('<.*?>')
  return re.sub(clean, '', text)

def preprocess_events(events):
  """ construct dictionary from event data """
  """ TODO: We need to add more attributes, Wenyu will investigate on this """
  return [
    {
      "title": event['title'],
      "location": event['location'],
      "description": strip_html(event['description']),
      "date": event['date']
    }
    for event in events
  ]

# need to slice this otherwise will exceed quota limits
preprocessed_data = preprocess_events(data)
preprocessed_data[0]

# write this to csv file
keys = preprocessed_data[0].keys()
with open('tamu_events.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(preprocessed_data)

# define columns we want in the embeddings and which one we want in metadata
columns_to_embed = ["title","description"]
columns_to_metadata = ["title", "location", "description", "date"]

docs = []
with open('tamu_events.csv', newline="", encoding='utf-8-sig') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for i, row in enumerate(csv_reader):
      to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
      values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
      to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
      newDoc = Document(page_content=to_embed, metadata=to_metadata)
      docs.append(newDoc)

print(docs[0])

# split the document using Chracter splitting.
splitter = CharacterTextSplitter(separator = "\n",
                                chunk_size = 1000,
                                chunk_overlap = 0,
                                length_function = len)
documents = splitter.split_documents(docs)

# generate embeddings from documents and store in a vector database
embeddings_model = get_embedding_function()
db = Chroma.from_documents(documents=documents, embedding=embeddings_model, persist_directory=CHROMA_PATH)
db.persist()