#from Pixegami's repository here: https://github.com/pixegami/rag-tutorial-v2

from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings