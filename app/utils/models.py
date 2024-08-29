import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from typing import List

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_response_farmacias(coordinates, docs, model):

    # Define the structure for the answer
    class Coordinates(BaseModel):
        latitude: float
        longitude: float

    class Pharmacies(BaseModel):
        name: str
        address: str = None  # Hacer la dirección opcional para "Tu Ubicación"
        coordinates: Coordinates

    class AnswerParser(BaseModel):
        pharmacies: List[Pharmacies]

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    # Instructions
    - Answer in Spanish.
    - Use local chilean time.
    - Provide a JSON list of the nearest and open pharmacies with their names, addresses, and coordinates.
    - Order the pharmacies from the closest to the farthest based on their proximity to the given coordinates.
    - Use 'local_lat' as the latitude and 'local_lng' as the longitude when determining proximity.


    Question: 
    Get the 3 nearest places to these coordinates: 
        lat: {lat}
        lng: {lng}

    Context: 
    {context}

    Answer:""")

    # Convert coordinates to float and split
    lat, lng = coordinates

    # Initialize the language model with the specified parameters
    model = ChatOpenAI(model=model, temperature=0.1, api_key=OPENAI_API_KEY)

    parser = JsonOutputParser(pydantic_object=AnswerParser)

    # Create the RAG chain
    rag_chain = prompt | model | parser

    # Create the context
    context = " ".join(docs)  # Combina todos los documentos en un solo contexto

    # Generate the response
    response = rag_chain.invoke({'lat': lat, 'lng': lng, 'context': context})

    final_response = {
        "user_location": {
            "latitud": lat,
            "longitud": lng
        },
        "farmacias": response  # Assuming response is the list of farmacias
    }

    return final_response