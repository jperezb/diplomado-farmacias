import os
import requests
import logging
import streamlit as st


CHATBOT_URL = os.getenv("CHATBOT_URL")
API_TOKEN = os.getenv("API_TOKEN")


def call_agent(prompt: str):
  headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_TOKEN}'
  }
  data = { "uuid-client": st.session_state['uuid-client'], "pregunta": prompt }

  response = None
  
  try:
    response = requests.get(CHATBOT_URL, params=data, headers=headers)
    logging.info(response)
    if response.status_code == 200:
      return response.json()
    else:
      return None
  except Exception as e:
    logging.error(e)
    logging.error(f"Error: {response.status_code}")
    logging.error(response.text)

  return

def call_agent_history(uuid: str):
  return None


def call_response():
  return {
    "uuid-client": "c1b9b1e0-1f1b-11e7-93ae-92361f002671",
    "response": [
      {
        "order": 1,
        "type": "map",
        "content": {
            "user_location": {
              "latitud": -33.4268151,
              "longitud": -70.5922854
            },
            "farmacias": [
                {
                    "nombre": "URGENCIA - AHUMADA",
                    "direccion": "LOS LEONES 1160",
                    "coordenadas": {
                        "lat": -33.429951,
                        "lng": -70.602705
                    },
                    "funcionamiento_hora_apertura": "00:00:00",
                    "funcionamiento_hora_cierre": "23:59:00",
                    "distancia": 1.03
                },
                {
                    "nombre": "URGENCIA - AHUMADA ",
                    "direccion": "CRISTOBAL COLON  5090",
                    "coordenadas": {
                        "lat": -33.41878,
                        "lng": -70.572098
                    },
                    "funcionamiento_hora_apertura": "00:00:00",
                    "funcionamiento_hora_cierre": "23:59:00",
                    "distancia": 2.08
                },
                {
                    "nombre": "URGENCIA - SALCOBRAND",
                    "direccion": "MANUEL MONTT 1140",
                    "coordenadas": {
                        "lat": -33.438335,
                        "lng": -70.615679
                    },
                    "funcionamiento_hora_apertura": "00:00:00",
                    "funcionamiento_hora_cierre": "23:59:00",
                    "distancia": 2.52
                }
            ]
        }
      }
    ]
  }