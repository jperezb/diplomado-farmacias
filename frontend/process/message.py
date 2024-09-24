from collections.abc import ItemsView
import streamlit as st
from streamlit import session_state as ss
from streamlit_folium import st_folium, folium_static
import folium
import logging
import process.history as History
import random
import string


def folium_map(data: dict):
  map = folium.Map(location=[
      data["user_location"]["latitud"], data["user_location"]["longitud"]
  ],
                   zoom_start=13)

  folium.Marker(location=[
      data["user_location"]["latitud"], data["user_location"]["longitud"]
  ],
                icon=folium.Icon(color="red")).add_to(map)

  for farmacia in data["farmacias"]:
    folium.Marker(
        location=[
            farmacia["coordenadas"]["lat"], farmacia["coordenadas"]["lng"]
        ],
        popup=
        f'{farmacia["nombre"]}\n{farmacia["direccion"]}\n{farmacia["distancia"]} km\n{farmacia["funcionamiento_hora_apertura"]} - {farmacia["funcionamiento_hora_cierre"]}',
        tooltip=farmacia["nombre"],
        icon=folium.Icon(color="blue")).add_to(map)

  return map


def map_directions(message):
  ul = message["user_location"]
  base = f'https://www.google.com/maps/dir/?api=1&origin={ul["latitud"]},{ul["longitud"]}&destination='

  directions = "Aqui tienes la rutas, a traves de Google Maps, para llegar a cada Farmacia:\n"

  for i in message["farmacias"]:
    coor = i['coordenadas']
    directions += f'- [{i["nombre"]}: {i["direccion"]}]({base}{coor["lat"]},{coor["lng"]})\n'

  return directions


def process(st, message):
  try:
    data = sorted(message['response'], key=lambda x: x['order'])

    for item in data:
      with st.chat_message('assistant'):
        if item['type'] == 'text':
          st.markdown(item['content'])
          History.add({
              "role": "assistant",
              "type": "text",
              "output": item['content']
          })
        elif item['type'] == 'map':
          key = ''.join(random.sample(string.ascii_letters, 20))
          folium_static(folium_map(item['content']))
          History.add({
              "role": "assistant",
              "type": "map",
              "key": key,
              "output": item['content']
          })

          info = map_directions(item['content'])
          st.markdown(info)

          History.add({"role": "assistant", "type": "text", "output": info})
  except Exception as e:
    logging.error(e, exc_info=True)

def add(st, item):
  with st.chat_message(item['role']):
    if item['type'] == 'text':
      st.markdown(item['output'])
      History.add({
          "role": item['role'],
          "type": "text",
          "output": item['output']
      })

def state_messages():
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      if message['type'] == 'text':
        st.markdown(message['output'])
      elif message['type'] == 'map':
        folium_static(folium_map(message['output']))
      elif message['type'] == 'text_map':
        st.markdown(message['output'])
