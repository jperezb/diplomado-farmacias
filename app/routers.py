import mimetypes
import os
import json
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from pydantic import BaseModel, Field
from app.utils.minsal import obtener_datos_locales_turnos
from app.utils.rag import get_documents, get_retriever
from app.utils.models import generate_response

# Crear un router
router = APIRouter()

# Obtiene los datos de Qdrant
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# AWS S3
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

MODEL_OPENAI = os.getenv("MODEL_OPENAI")

class QueryRequest(BaseModel):
    input: str

@router.post("/locales-turnos/")
async def obtener_locales_turnos(question: QueryRequest):
    """API endpoint para obtener los datos de locales de turnos."""
    try:
        datos = obtener_datos_locales_turnos()
        documents = get_documents(datos)
        # retriever = get_retriever(documents)

        # Generate Response
        response = generate_response(question.input, documents, MODEL_OPENAI)

        return JSONResponse(content=response)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"message": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@router.post("/upload/")
async def upload_file(metadata: str = Form(...), file: UploadFile = File(...)):
    """API endpoint para subir un archivo."""
    print("***** Inicia la subida del archivo *****")
    try:
        # Convertir metadata de cadena JSON a diccionario
        import json
        metadata_dict = json.loads(metadata)

        # Detectar el tipo MIME del archivo
        mime_type, _ = mimetypes.guess_type(file.filename)
        mime_type = mime_type or 'application/octet-stream'

        # Procesar embeddings y verificar duplicados
        if mime_type == 'application/pdf':
            embedding = make_embedding_pdf(file, metadata_dict)
        else: 
            embedding = make_embedding(file, mime_type, metadata_dict)

        if embedding["status"] == False:
            return JSONResponse(content=embedding)

        ### Subir el archivo a S3
        ### Este módulo se puede comentar si no se cuenta con S3
        success = upload_file_to_s3(file, AWS_S3_BUCKET)
        if not success:
            raise HTTPException(status_code=500, detail="Error al subir el archivo a S3.")
        ### 

        return JSONResponse(content={"filename": file.filename, "message": "Archivo subido exitosamente a S3."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@router.post("/query/")
async def query(request: QueryRequest):

    class ReferencesParser(BaseModel):
        document: str = Field(..., description="Document name")
        page: str = Field(..., description="Document page")

    class AnswerParser(BaseModel):
        answer: str = Field(..., description="Answer for the user's question.")
        references: List[ReferencesParser] = Field(..., description="List of references included in the answer.")
        
    # Convierte los documentos en embeddings y los guarda en Qdrant
    embeddings = OpenAIEmbeddings()

    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        retrieval_mode=RetrievalMode.DENSE,
    )

    # Se crea el retriever
    retriever = qdrant.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use only the following pieces of retrieved context to answer "
        "the question. If the answer is not in the documents, say that you "
        "don't know using different sentences in friendly mode. Use three sentences maximum and keep the "
        "answer concise. "
        "Add the document references in your answer "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    parser = PydanticToolsParser(pydantic_object=[AnswerParser])

    model = ChatOpenAI(model="gpt-4o-mini")

    question_answer_chain = create_stuff_documents_chain(model, prompt, parser)
    print("** question_answer_chain **")
    print(question_answer_chain)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("** rag_chain **")
    print(rag_chain)
    result = rag_chain.invoke({"input": request.input})

    # return JSONResponse(content={"result": result})

    return {"result": result}

@router.post("/files/delete")
async def files_delete(request: QueryRequest):
    """ Elimina los points del vector store y un archivo desde S3 """

    filename = request.input

    # Se eliminan los points en el vector store
    points_deleted = delete_documents_by_filename(QDRANT_COLLECTION_NAME, filename)

    # Si se eliminaron, se debe eliminar el archivo en S3
    if points_deleted:
        s3_deleted = delete_file_from_s3(AWS_S3_BUCKET, filename)
        
        if s3_deleted:
            return {"result": "Delete process OK."}
    else:
        return {"result": "Nothing to delete."}