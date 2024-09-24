import streamlit as st
from tinydb import TinyDB, Query
import os

db = TinyDB('db/user-session.json')

def load():
  #st.session_state.login = False
  data = db.all()
  for dictionary in data:
    #if "user" in dictionary:
    #  st.session_state.login = dictionary['login']
    if "uuid" in dictionary:
      if os.getenv('VALIDATE_TOS') and 'uuid-client' in st.session_state:
        tos_id = st.session_state['uuid-client'] + '_TOS'
        st.session_state[tos_id] = True

def accept_tos():
  tos_id = st.session_state['uuid-client'] + '_TOS'
  st.session_state[tos_id] = True
  db.insert({
    "ToS": True,
    "uuid": st.session_state['uuid-client']
  })

def login():
  #db.insert({
  #  'user': st.session_state['uuid-client'],
  #  'login': True
  #})
  st.session_state.login = True

def logout():
  #Q = Query()
  #if 'uuid-client' in st.session_state:
  #  db.remove(Q.user == st.session_state['uuid-client'])
  st.session_state.login = False
  st.session_state["authentication_status"] = False

def clear():
  db.truncate()