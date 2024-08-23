import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_response(question, docs, model):
    """
    Generate a response for a given question using retrieved context.

    This function creates a prompt for a question-answering task, invokes the RAG chain to get the response,
    and measures the time taken to generate the response.

    Parameters:
    question (str): The question to be answered.
    context (str): The context to be used for answering the question.

    Returns:
    str: The generated response.
    """

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    # Instructions
    - Answer in spanish

    Question: {question}

    Context: {context}

    Answer:""")

    # Initialize the language model with the specified parameters
    model = ChatOpenAI(model=model, temperature=0.1, api_key=OPENAI_API_KEY)

    # Create the RAG chain
    rag_chain = prompt | model | StrOutputParser()

    # Create the context
    context = ""
    for doc in docs:
        print(doc)  # Imprime el contenido de cada documento
        context += doc + " "  # Agrega el contenido de cada documento al contexto

    # Generate the response
    response = rag_chain.invoke({'question': question, 'context': context})

    return response