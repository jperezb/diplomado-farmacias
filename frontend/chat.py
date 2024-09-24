import streamlit as st
from streamlit import session_state as ss
import logging
import process.api as api
import process.message as message
import process.history as History
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import process.session as Session


st.markdown("""
    <style>
    .reportview-container {
    margin-top: -2em;
    }
    [data-testid="stSidebarCollapsedControl"] {
    display: none
    }
    #MainMenu {visibility: hidden;}
    .stAppDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    </style>
    """,
    unsafe_allow_html=True)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_FILENAME = './config/access.yml'

with open(CONFIG_FILENAME) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(config['credentials'],
                                    config['cookie']['name'],
                                    config['cookie']['key'],
                                    config['cookie']['expiry_days'],
                                    config['pre-authorized'])

authenticator.login(location='unrendered')

Session.load()
if not st.session_state["authentication_status"]:
    st.switch_page('./pages/account.py')

with st.header(body="Dr. Igualin"):
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button('Logout'):
            Session.logout()
            st.switch_page('./pages/account.py')
    with cc2:    
        if st.button('Borrar Historial?'):
            History.clear()
    

History.load()

def get_history(st):
    response = api.call_agent_history(ss['uuid-client'])
    if response:
        st.write('not yet')
    else:                
        st.chat_message("assistant").markdown(
                """Ha ocurrido un error, intenta de nuevo.""")

def search_for_answer(prompt: str):
    with st.spinner("Buscando la respuesta..."):
        response = api.call_agent(prompt)

        if response:
            message.process(st, response)
        else:                
            st.chat_message("assistant").markdown(
                """😭 Ha ocurrido un error, intenta de nuevo. 😭""")

col1, col2 = st.columns(2)
with col1:
    st.title("Dr. Igualín Chatbot")
with col2:
    st.image('assets/dr-igualin.png', width=100)


st.info("Pregúntame todo sobre farmacias de turno en Chile."
        " Si tienes duda con algún médicamente, no dudes en preguntar.")

#Paint the chat history in the session_state
message.state_messages()
 
if prompt := st.chat_input("Que te gustaría saber?"):
    st.chat_message("user").markdown(prompt)

    History.add({
        "role": "user",
        "type": "text",
        "output": prompt
    })
    search_for_answer(prompt)
