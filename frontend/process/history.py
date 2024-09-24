import streamlit as st
from tinydb import TinyDB

db = TinyDB('db/history.json')

def load():
  if "messages" not in st.session_state:
    st.session_state.messages = []
  st.session_state.messages = db.all()

def add(message):
  db.insert(message)
  st.session_state.messages.append(message)

def clear():
  db.truncate()