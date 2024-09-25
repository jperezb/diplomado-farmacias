import os
import functools
import operator
import logging
import json
import shutil
from typing import Dict, Any, List, Optional, Annotated, Sequence, TypedDict, Literal
from datetime import datetime
from app.utils.minsal import obtener_datos_locales_turnos
from app.utils.rag import get_documents
from app.utils.locations import get_coordinates, farmacias_cercanas
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import create_model, BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.responses import JSONResponse
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_message_histories import RedisChatMessageHistory

# Configuración de claves API
load_dotenv()

langchain_api_key = os.getenv('langchain_api_key')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_api_key = os.getenv('openai_api_key')

tavily_api_key = os.getenv('tavily_api_key')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Redis
REDIS_URL = os.getenv('REDIS_URL')

# LOGS
class QueryRequest(BaseModel):
    input: str

log_file_path = 'doc/log_farmacias_turno.txt'

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

def format_history(messages):
    formatted_history = ''
    for message in messages:
        role = 'Usuario' if message.type == 'human' else 'Asistente'
        formatted_history += f'{role}: {message.content}\n'
    return formatted_history

# Obtener todos los mensajes del historial
HISTORY = RedisChatMessageHistory(session_id = 'sesionDiplomado', url=REDIS_URL).messages 
formatted_history = format_history(HISTORY)        

# Inicialización de FastAPI
app = FastAPI()

# Cargar y procesar documentos del vademecum
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

######
# funciones framacias

def get_farmacias_turno(question: QueryRequest):
    """API endpoint para obtener los datos de locales de turnos."""
    try:

        # Función para escribir comentarios en el archivo de log
        print(f"get_farmacias_turno question: ",question)
        coordinates = get_coordinates(question.input)
        print(f"get_farmacias_turno coordinates: ",coordinates)

        # Obtener la fecha formateada
        fecha_formateada = datetime.now().strftime("%Y-%m-%d")
        directorio = "files/"
        file_path = f"{directorio}{fecha_formateada}.txt"

        if os.path.exists(file_path):
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
            with open(file_path, "w", encoding='utf-8') as file:
                file.write(datos_json)

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

    # Definir el prompt
    prompt_qd = ChatPromptTemplate.from_template("""
    Eres un asistente que tiene como tarea dar respuesta a preguntas de medicamentos.
    Utilice las siguientes piezas de contexto recuperado para responder la pregunta.
    Si se te pide buscar medicamentos alternativos o similares a un medicamento específico, usa los que tengan mismas Indications y entrega la respuesta, por ejemplo si te preguntan que medicamento tiene indicaciones similar  a Omeprazole, debes responder Esomeprazole , ya que sus indations son las mismas Acid Reflux.

    Si te preguntan por un medicamento especifico para una enfermedad específica o indicaciones específicas, di que no puedes responder esa pregunta por política, como ejemplo toma, que si te preguntan que medicamente sirve para la enfermedad o indicaciones de depresión, debes respoter no está en tu política recomendar medicamentos para enfermedades específicas.
    Si no sabe la respuesta, simplemente diga que no la sabe.

    Utilice tres oraciones como máximo y mantenga la respuesta concisa.

    # Instrucciones
    - Responde siempre en español, sólo usa la información entregada en Contexto para tu respuesta

    **Pregunta** 
    {question}

    **Historial**
    {history}
                                                
    **Contexto**
    {context}

    **Respuesta**
    """)

    try:
        if not medication.strip():
            raise ValueError("El nombre del medicamento está vacío")
        
        try:
            relevant_docs = retriever.invoke(medication)
        except Exception as retriever_error:
            logging.error(f"Error al obtener documentos relevantes para {medication}: {str(retriever_error)}")
            return HumanMessage(content=f"Lo siento, hubo un problema al buscar información sobre {medication}. Por favor, intente nuevamente más tarde.")
        
        if not relevant_docs:
            logging.warning(f"No se encontraron documentos relevantes para el medicamento: {medication}")
            return HumanMessage(content=f"Lo siento, no se encontró información para el medicamento: {medication}")

        # Convertir el contexto en una cadena de texto
        context_text = " ".join([doc.page_content for doc in relevant_docs])

        # Formatear el historial
        formatted_history = format_history(HISTORY)        
        
        # Ejecutar la cadena y obtener la respuesta
        try:
            print("****** medication3 ******")
            print(medication)
            print("****** !medication3 ******")
            response = prompt_qd.invoke({"question": medication, "context": context_text, "history": formatted_history})
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
    write_log(f"entro a get_last_address : address = {address}")
    return True


# Define tool options
options = ["FINISH", "get_address_info", "get_medication_info", "get_last_address"]

# Create a dynamic model using `create_model` from Pydantic
routeResponse = create_model(
    "routeResponse",
    next=(Literal[tuple(options)], ...)  # Unpack the options into Literal
)

# List of members (tools)
members = ["get_address_info", "get_medication_info", "get_last_address"]

system_prompt = ("""Eres un supervisor encargado de gestionar una conversación entre las siguientes tools: {members}. 
    Dada la solicitud del usuario a continuación, responde con el tool que debe actuar a continuación. Cada tool realizará una tarea y responderá con sus resultados y estado.
    Cuando termines, responde con FINISH.

    Instrucciones para elegir el próximo tool:

    1. **Si la pregunta contiene una dirección específica o detalles de ubicación** (por ejemplo, una dirección de calle o nombre de edificio), selecciona 'get_address_info'. 
    Usa esta opción solo cuando la entrada se refiera explícitamente a una ubicación, dirección o instrucciones de cómo llegar.

    2. **Si la pregunta está relacionada con cualquier aspecto o dimensión de los medicamentos** (por ejemplo, alternativas de medicamentos, efectos secundarios, indicaciones, dosificación, interacciones, etc.), selecciona 'get_medication_info'. 
    Esto incluye cualquier contexto médico relacionado con el propósito de un medicamento, administración, efectos secundarios, contraindicaciones u otros detalles farmacológicos. Responde en español.

    3. **Para cualquier otro tipo de consulta** que no caiga en las dos categorías anteriores, selecciona 'get_last_address'. 
    Esto incluye cualquier pregunta fuera del ámbito de medicamentos o direcciones, como consultas sobre la capital de un país, matemáticas o cualquier otro campo no relacionado. 

    Si no entiendes la pregunta, solicita más información indicando: 'Necesito más información para poder asistirte.'
    Recuerda, eres un asistente especializado en proporcionar información sobre medicamentos, sus diversas dimensiones médicas, y las ubicaciones de farmacias en base a una dirección dada. 
    Debes asistir solo dentro de estos contextos específicos.

    Todas las respuestas y comunicaciones deben ser en español.
    """)

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
        # print(f"agent_node result : ",result)

        if name == 'get_address_info':
            try:
                if len(result["messages"]) >= 3:
                    # Extraer el ToolMessage directamente del resultado, asumiendo que el tercer mensaje existe
                    tool_message = result["messages"][2]  # Índice 2 porque es el tercer elemento
                    print(f"agent_node tool_message=result.mesage[2]: {tool_message}")
                    return {"messages": [HumanMessage(content=tool_message.content, name=tool_message.name)]}
                else:
                    # Si no hay suficientes mensajes, devolver un error adecuado
                    print(f"Error: No hay suficientes mensajes en el resultado. Mensajes recibidos: {len(result['messages'])}")
                    return {"messages": [HumanMessage(content="No se encontró suficiente información para procesar la dirección.", name=name)]}
        
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

    # Crear o cargar el historial de la conversación
    history = RedisChatMessageHistory(session_id = 'sesionDiplomado', url=REDIS_URL)

    # Agregar el mensaje del usuario al historial
    history.add_user_message(pregunta)

    # Obtener todos los mensajes del historial
    HISTORY = history.messages 

    final_response = None  # Variable para almacenar la respuesta final

    # Ejecuta el flujo de trabajo con la pregunta
    for s in graph.stream(     
        {"messages": [HumanMessage(content=pregunta), SystemMessage(content=format_history(HISTORY))]},
        {"recursion_limit": 10}):

        if "__end__" not in s:
            print("********** S **********")
            print(s)
            print("********** !S **********")

            # Almacena la respuesta final de cada herramienta
            if 'messages' in s.get('get_address_info', {}):
                get_address_info_response = s['get_address_info']['messages'][0].content
#                write_log(f"get_address_info_response en final_response {get_address_info_response}")
                final_response=get_address_info_response
                try:
                    desescaped_content = json.loads(final_response)
                    resultado = {
                    "uuid-client": "c1b9b1e0-1f1b-11e7-93ae-92361f002671",
                    "response": [
                        {
                            "order": 1,
                            "type": "map",
                            "content": desescaped_content
                        }
                    ]
                    }
                except:
                    desescaped_content =final_response
                    resultado = {
                    "uuid-client": "c1b9b1e0-1f1b-11e7-93ae-92361f002671",
                    "response": [
                        {
                            "order": 1,
                            "type": "text",
                            "content": desescaped_content
                        }
                    ]
                    }

                final_response = resultado

            elif 'messages' in s.get('get_medication_info', {}):
                final_response = s['get_medication_info']['messages'][0].content

                resultado = {
                    "uuid-client": "c1b9b1e0-1f1b-11e7-93ae-92361f002671",
                    "response": [
                        {
                            "order": 1,
                            "type": "text",
                            "content": final_response
                        }
                    ]
                }
                final_response = resultado

            elif 'messages' in s.get('get_last_address', {}):
                final_response =  s['get_last_address']['messages'][0].content

                resultado = {
                    "uuid-client": "c1b9b1e0-1f1b-11e7-93ae-92361f002671",
                    "response": [
                        {
                            "order": 1,
                            "type": "text",
                            "content": final_response
                        }
                    ]
                }

                final_response = resultado

    if final_response:
        # Si el resultado es un diccionario, lo convertimos en un array con ese diccionario dentro
        if isinstance(final_response, dict):
            print("final_response es un diccionario")
#            final_response = [final_response]  # Convertir el diccionario en una lista de un solo objeto
            final_response = final_response  # Convertir el diccionario en una lista de un solo objeto

            history_message = str(final_response['response'][0]['content'])

        else:
            print("final_response es una cadena JSON")
            
            try:
                # Verificar si la cadena no está vacía antes de intentar decodificarla
                if final_response.strip():  # Verificar que no esté vacía
                    # Intentamos cargarlo como JSON y luego asegurarnos de que sea una lista
                    final_response = json.loads(final_response)
                    if isinstance(final_response, dict):
                        final_response = final_response  # Convertir el diccionario en una lista
                    elif not isinstance(final_response, list):
                        raise ValueError("final_response no es un array o diccionario válido")
                else:
                    # Si la cadena está vacía, lanzar una excepción o manejar el caso adecuadamente
                    raise ValueError("final_response está vacío o no es JSON válido")

                history_message = final_response['response'][0]['content']

            except json.JSONDecodeError as e:
                print(f"Error al decodificar JSON: {e}")
                return JSONResponse(content={"error": "Formato JSON inválido"}, status_code=400)
            except ValueError as e:
                print(f"Error: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=400)

        # Devolver el array de objetos
        #return JSONResponse(content=final_response, status_code=200)

        # Agregar la respuesta del AI al historial
        history.add_ai_message(history_message)

        return final_response
    else:
        # Manejar el caso donde final_response es None o vacío
        return JSONResponse(content={"error": "final_response es None o vacío"}, status_code=400)