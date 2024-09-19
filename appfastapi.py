from dotenv import load_dotenv

from fastapi import FastAPI
import functools
from typing import Annotated, Sequence, TypedDict, Literal
from pydantic import create_model
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import operator


#farm

from typing import Dict, Any
import logging
import mimetypes
import os
import json
import shutil
from typing import List
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.chains import create_retrieval_chain
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader 
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import math
import urllib.parse
import urllib.request
import re
import logging
from typing import Optional





from app.utils.minsal import obtener_datos_locales_turnos
from app.utils.rag import get_documents, get_retriever
#from app.utils.models import generate_response_farmacias
from app.utils.locations import get_coordinates, farmacias_cercanas, calcular_distancia


from pydantic import BaseModel, Field
class QueryRequest(BaseModel):
    input: str
log_file_path = 'doc/log_farmacias_turno.txt'
def write_log00(message: str):

    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(message + '\n')


import os

def write_log(message: str):
    try:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(message + '\n')
    except FileNotFoundError:
        # Si el archivo no existe, creamos el directorio y el archivo
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(message + '\n')
    except Exception as e:
        print(f"Error al escribir en el archivo de log: {e}")



# Configuración de claves API

load_dotenv()

langchain_api_key = os.getenv('langchain_api_key')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_api_key = os.getenv('openai_api_key')

tavily_api_key = os.getenv('tavily_api_key')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')



# Inicialización de FastAPI
app = FastAPI()

# Cargar y procesar documentos
file_path = 'doc/RESUMEN medicamentos.txt'
loader = TextLoader(file_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["/n"])
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Configurar Qdrant


qdrant = QdrantVectorStore.from_documents(
    docs, embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,
    collection_name="medicamentosv3.1",
    force_recreate=True,
)

retriever = qdrant.as_retriever()

# Configuración del LLM y el prompt
llmqd = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# Definir el prompt
prompt_qd = ChatPromptTemplate.from_template("""
Eres un asistente que tiene como tarea dar respuesta a preguntas de medicamentos.
Utilice las siguientes piezas de contexto recuperado para responder la pregunta.
Si se te  pide buscar medicamentos alternativos o similares a un medicamento específico, usa los que tengan mismas Indications y entrega la respuesta, por ejemplo si te preguntan que medicamento tiene indicasiones similar  a Omeprazole, debes responder Esomeprazole , ya que sus indations son las mismas Acid Reflux.

Si te preguntan por un medicamento especifico para una enfermedad específica o indations específica, di que no puedes responder esa pregunta por política ,  como ejemplo toma, que si te preguntan que medicamente sirve para la enfermedad o indicaciones de depresión , debes respoter no está en tu política recomendar medicamentos para enfermedades específicas.
Si no sabe la respuesta, simplemente diga que no la sabe.

Utilice tres oraciones como máximo y mantenga la respuesta concisa.

# Instrucciones
- Responde siempre en español, sólo usa la información entregada en Contexto para tu respuesta
  

Pregunta: {question}
Contexto: {context}
Respuesta:""")


######
# funciones framacias



def get_farmacias_turno(question: QueryRequest):
    """API endpoint para obtener los datos de locales de turnos."""
    try:

        # Función para escribir comentarios en el archivo de log

        print(f"get_farmacias_turno question: ",question)
        coordinates = get_coordinates(question.input)
        print(f"get_farmacias_turno coordinates: ",coordinates)

        directorio='doc'
        file_path = 'doc/2024-08-31.txt'
        if os.path.exists(file_path):
            #print(f"El archivo {file_path} ya existe. No se llamará a obtener_datos_locales_turnos.")
            # Si el archivo existe, cargar los datos desde el archivo
            with open(file_path, "r", encoding='utf-8') as file:
                datos_json = file.read()
            datos = json.loads(datos_json)
            print(f"get_farmacias_turno datos if path exist  json.loads :::: ",len(datos))
        else:
            # Vaciar el directorio eliminándolo y recreándolo
            if os.path.exists(directorio):
                shutil.rmtree(directorio)
            os.makedirs(directorio)
            
            # Si el archivo no existe, obtener los datos y guardarlos en el archivo
            datos = obtener_datos_locales_turnos()
            print(f"get_farmacias_turno datos de obtener_datos_locales_turnos else :::: ",len(datos))

            # Convertir el objeto a una cadena JSON
            datos_json = json.dumps(datos, ensure_ascii=False, indent=4)

            # Guardar los datos en un archivo con el nombre basado en la fecha
            with open(archivo_nombre, "w", encoding='utf-8') as file:
                file.write(datos_json)
        documents = get_documents(datos)
        print(f"get_farmacias_turno documents :::: ",len(documents))
        response = farmacias_cercanas(coordinates, datos_json, 3)

        return response
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"message": 'excepcion'+ e.detail})



###########






# Define tools
@tool
def get_address_info(address01: str) -> Optional[HumanMessage]:
    """Obtiene información sobre la dirección que se consulta explícitamente."""
    try:
        if not address01.strip():
            raise ValueError("La dirección proporcionada está vacía")

        last_address_queried = address01

        # Corregir la clase QueryRequest y su uso
        query2 = QueryRequest(input=last_address_queried)
        
        try:
            respuesta = get_farmacias_turno(query2)
        except Exception as turno_error:
            logging.error(f"Error al obtener farmacias de turno para {address01}: {str(turno_error)}")
            return HumanMessage(content=f"Lo siento, hubo un problema al buscar farmacias de turno para la dirección: {address01}. Por favor, intente nuevamente más tarde.")

        if not respuesta:
            logging.warning(f"No se encontró información de farmacias para la dirección: {address01}")
            return HumanMessage(content=f"Lo siento, no se encontró información de farmacias para la dirección: {address01}")

        return respuesta

    except ValueError as ve:
        error_msg = f"Error de valor en get_address_info: {str(ve)}"
        logging.error(error_msg)
        return HumanMessage(content=f"Error: {error_msg}")

    except Exception as e:
        error_msg = f"Error inesperado en get_address_info: {str(e)}"
        logging.exception(error_msg)  # This logs the full stack trace
        return HumanMessage(content="Lo siento, ocurrió un error inesperado al procesar su solicitud de información de dirección.")

@tool
def get_medication_info(medication: str) -> Optional[HumanMessage]:
    """Obtiene información sobre un medicamento."""
    try:
        if not medication.strip():
            raise ValueError("El nombre del medicamento está vacío")
        
        try:
            relevant_docs = retriever.get_relevant_documents(medication)
        except Exception as retriever_error:
            logging.error(f"Error al obtener documentos relevantes para {medication}: {str(retriever_error)}")
            return HumanMessage(content=f"Lo siento, hubo un problema al buscar información sobre {medication}. Por favor, intente nuevamente más tarde.")
        
        if not relevant_docs:
            logging.warning(f"No se encontraron documentos relevantes para el medicamento: {medication}")
            return HumanMessage(content=f"Lo siento, no se encontró información para el medicamento: {medication}")

        # Convertir el contexto en una cadena de texto
        context_text = " ".join([doc.page_content for doc in relevant_docs])
        
        # Ejecutar la cadena y obtener la respuesta
        try:
            response = prompt_qd.invoke({"question": medication, "context": context_text})
        except Exception as prompt_error:
            logging.error(f"Error al procesar la respuesta para {medication}: {str(prompt_error)}")
            return HumanMessage(content=f"Lo siento, hubo un problema al procesar la información sobre {medication}. Por favor, intente nuevamente.")

        if not response:
            raise ValueError(f"No se pudo generar una respuesta para el medicamento: {medication}")

        return response

    except ValueError as ve:
        error_msg = f"Error de valor en get_medication_info: {str(ve)}"
        logging.error(error_msg)
        return HumanMessage(content=f"Error: {error_msg}")

    except Exception as e:
        error_msg = f"Error inesperado en get_medication_info: {str(e)}"
        logging.exception(error_msg)  # This logs the full stack trace
        return HumanMessage(content="Lo siento, ocurrió un error inesperado al procesar su solicitud.")


@tool
def get_last_address(address: str) -> str:
    """Indica que la consulta no está dentro del ámbito de asistencia."""
    write_log(f"entro a  get_last_address : address = {address}")
    return f"la consulta  {address}  no está en el contexto de mi asistencia , favor replantee su consulta"


# Define tool options
options = ["FINISH", "get_address_info", "get_medication_info", "get_last_address"]

# Create a dynamic model using `create_model` from Pydantic
routeResponse = create_model(
    "routeResponse",
    next=(Literal[tuple(options)], ...)  # Unpack the options into Literal
)

# List of members (tools)
members = ["get_address_info", "get_medication_info", "get_last_address"]




system_prompt = (
    "Eres un supervisor encargado de gestionar una conversación entre los siguientes trabajadores: {members}. "
    "Dada la solicitud del usuario a continuación, responde con el trabajador que debe actuar a continuación. Cada trabajador realizará una tarea y responderá con sus resultados y estado. "
    "Cuando termines, responde con FINISH."
    "\n\n"
    "Instrucciones para elegir el próximo trabajador:"
    "\n\n"
    "1. **Si la pregunta contiene una dirección específica o detalles de ubicación** (por ejemplo, una dirección de calle o nombre de edificio), selecciona 'get_address_info'. "
    "Usa esta opción solo cuando la entrada se refiera explícitamente a una ubicación, dirección o instrucciones de cómo llegar."
    "\n\n"
    "2. **Si la pregunta está relacionada con cualquier aspecto o dimensión de los medicamentos** (por ejemplo, alternativas de medicamentos, efectos secundarios, indicaciones, dosificación, interacciones, etc.), selecciona 'get_medication_info'. "
    "Esto incluye cualquier contexto médico relacionado con el propósito de un medicamento, administración, efectos secundarios, contraindicaciones u otros detalles farmacológicos."
    "\n\n"
    "3. **Para cualquier otro tipo de consulta** que no caiga en las dos categorías anteriores, selecciona 'get_last_address'. "
    "Esto incluye cualquier pregunta fuera del ámbito de medicamentos o direcciones, como consultas sobre la capital de un país, matemáticas, historia o cualquier otro campo no relacionado. "
    "En este caso, simplemente indica: 'Esta consulta no está dentro del ámbito de mi asistencia.' y no proceses la solicitud más allá de esta respuesta."
    "\n\n"
    "Si no entiendes la pregunta, solicita más información indicando: 'Necesito más información para poder asistirte.' "
    "Recuerda, eres un asistente especializado en proporcionar información sobre medicamentos, sus diversas dimensiones médicas, y las ubicaciones de farmacias en base a una dirección dada. "
    "Debes asistir solo dentro de estos contextos específicos."
    "\n\n"
    "Todas las respuestas y comunicaciones deben ser en español."
)



# Define supervisor prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# Initialize the model
llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)




def supervisor_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        supervisor_chain = (
            prompt
            | llm.with_structured_output(routeResponse)
        )
        result = supervisor_chain.invoke(state)
        return result

    except ValueError as ve:
        # Manejar errores específicos de validación o estructura
        error_msg = f"Error de validación en supervisor_agent: {str(ve)}"
        logging.error(error_msg)
        return {"error": error_msg, "type": "validation_error"}

    except TypeError as te:
        # Manejar errores de tipo, por ejemplo, si state no es del tipo esperado
        error_msg = f"Error de tipo en supervisor_agent: {str(te)}"
        logging.error(error_msg)
        return {"error": error_msg, "type": "type_error"}

    except Exception as e:
        # Capturar cualquier otra excepción no prevista
        error_msg = f"Error inesperado en supervisor_agent: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg, "type": "unexpected_error"}


# Define agent node function


def agent_node(state, agent, name):
    try:
        result = agent.invoke(state)

        if name == 'get_address_info':
            try:
                # Extraer el ToolMessage directamente del resultado, asumiendo que siempre es el tercer elemento
                tool_message = result["messages"][2]  # Índice 2 porque es el tercer elemento
                # Retornar el ToolMessage en el formato deseado
                return {"messages": [HumanMessage(content=tool_message.content, name=tool_message.name)]}
            except IndexError:
                write_log(f"Error: No se encontró el ToolMessage esperado en get_address_info. Resultado: {result}")
                return {"messages": [HumanMessage(content="Error al procesar la información de dirección.", name=name)]}
        else:
            try:
                return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}
            except IndexError:
                write_log(f"Error: No se encontraron mensajes en el resultado para {name}. Resultado: {result}")
                return {"messages": [HumanMessage(content=f"Error al procesar la información para {name}.", name=name)]}

    except Exception as e:
        error_message = f"Error en agent_node para {name}: {str(e)}"
        write_log(error_message)
        return {"messages": [HumanMessage(content=f"Ocurrió un error al procesar su solicitud para {name}.", name=name)]}



# Initialize tool agents
address_agent = create_react_agent(llm, tools=[get_address_info])
medication_agent = create_react_agent(llm, tools=[get_medication_info])
last_address_agent = create_react_agent(llm, tools=[get_last_address])

# Define agent nodes for each tool
address_node = functools.partial(agent_node, agent=address_agent, name="get_address_info")
medication_node = functools.partial(agent_node, agent=medication_agent, name="get_medication_info")
last_address_node = functools.partial(agent_node, agent=last_address_agent, name="get_last_address")

# Define workflow graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("get_address_info", address_node)
workflow.add_node("get_medication_info", medication_node)
workflow.add_node("get_last_address", last_address_node)
workflow.add_node("supervisor", supervisor_agent)

# Set up edges for graph workflow
for member in members:
    workflow.add_edge(member, "supervisor")

# Define the conditional edges for the supervisor to determine the next step
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Add entry point to the graph
workflow.add_edge(START, "supervisor")

# Compile the graph
graph = workflow.compile()


all_messages = []  # Lista para almacenar todos los mensajes

# Definir la ruta principal de FastAPI
@app.get("/")
def index(pregunta: str):
    # Usa la variable `pregunta` como entrada para el flujo de trabajo
    consulta1 = pregunta

    final_response = None  # Variable para almacenar la respuesta final

    # Ejecuta el flujo de trabajo con la pregunta
    for s in graph.stream(     
        {"messages": [HumanMessage(content=consulta1)]},
        {"recursion_limit": 100},

    ):


        if "__end__" not in s:
            # Almacena la respuesta final de cada herramienta
            if 'messages' in s.get('get_address_info', {}):
                get_address_info_response = s['get_address_info']['messages'][0].content
#                write_log(f"get_address_info_response en final_response {get_address_info_response}")
                final_response=get_address_info_response
                desescaped_content = json.loads(final_response)
                resultado = {
                    "uuid-client": "c1b9b1e0-1f1b-11e7-93ae-92361f002671",
                    "response": {
                        "order": 1,
                        "type": "map",
                    "content": [desescaped_content]}
                }
                final_response = resultado

            elif 'messages' in s.get('get_medication_info', {}):
                final_response = s['get_medication_info']['messages'][0].content
                resultado = {
                    "uuid-client": "c1b9b1e0-1f1b-11e7-93ae-92361f002671",
                    "response": {
                        "order": 1,
                        "type": "text",
                    "content": final_response}
                }
                final_response = resultado

            elif 'messages' in s.get('get_last_address', {}):
                final_response =  "Su consulta no está en el contexto o ámbito de mi asistencia , favor replantee su consulta."
                resultado = {
                    "uuid-client": "c1b9b1e0-1f1b-11e7-93ae-92361f002671",
                    "response": {
                        "order": 1,
                        "type": "text",
                    "content": final_response}
                }                
                final_response = resultado



    if final_response:
        if isinstance(final_response, dict):
            print("final_response es un diccionario")
        else:
            print("final_response es una cadena JSON")
            # Solo convertimos si no es ya un diccionario
            final_response = json.loads(final_response)
        
        #return JSONResponse(content=final_response, status_code=200)
        return JSONResponse(content=final_response)