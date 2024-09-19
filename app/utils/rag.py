import json
import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveJsonSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



# Function to Get the Documnets
def get_documents(data):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    print(f'OPENAI_API_KEY en get_documents::: ' ,OPENAI_API_KEY)
    # Convertir cada diccionario en un objeto Document
    documents = [
        Document(page_content=json.dumps(item), metadata={"source": "locales_turnos"})
        for item in data
    ]

    # # Generate and split documents into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    # texts = text_splitter.split_documents(documents)

    splitter = RecursiveJsonSplitter(max_chunk_size=300)
    texts = splitter.split_text(json_data=data, convert_lists=True)

    return texts


# Function to Get the Retriever
def get_retriever(texts, k=20):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    """
    Create a retriever object from the given texts.

    This function generates embeddings for the provided texts using OpenAI's API key,
    creates a vector store using FAISS, and returns a retriever object configured
    with the specified search parameters.

    Parameters:
    texts (list): A list of text documents to be embedded and stored.
    k (int): The number of nearest neighbors to retrieve. Default is 20.

    Returns:
    VectorStoreRetriever: A retriever object for searching the vector store.
    """
    # Generate embeddings for the texts using OpenAI's API key
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Create the vector store from the documents and embeddings
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Generate and return the retriever object with search parameters
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever