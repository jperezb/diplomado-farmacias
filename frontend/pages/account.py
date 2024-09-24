import streamlit as st
from streamlit import session_state as ss
import streamlit_authenticator as stauth
import yaml
import process.history as History
from yaml.loader import SafeLoader
import process.session as Session

CONFIG_FILENAME = './config/access.yml'

st.set_page_config(page_title="Login",
                   layout="wide",
                   initial_sidebar_state="collapsed")

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



col1, col2, col3 = st.columns(3)
with col2:
    st.image('assets/dr-igualin.png', width=200)
    st.header('Dr. Igualin')

with open(CONFIG_FILENAME) as file:
    config = yaml.load(file, Loader=SafeLoader)
    
authenticator = stauth.Authenticate(config['credentials'],
                                    config['cookie']['name'],
                                    config['cookie']['key'],
                                    config['cookie']['expiry_days'],
                                    config['pre-authorized'])

authenticator.login(location='main')

Session.load()

if ss["authentication_status"]:
    st.session_state['uuid-client'] = config['credentials']['usernames'][ss["username"]]['uuid']
    tos_id = config['credentials']['usernames'][ss["username"]]['uuid'] + '_TOS'
    
    Session.login()

    #History.clear()
    if tos_id not in ss or ss[tos_id] is False:
        st.switch_page('./pages/tos.py')
    else:
        st.switch_page('./chat.py')

elif ss["authentication_status"] is False:
    st.error('Usuario/password es incorrecto')
elif ss["authentication_status"] is None:
    st.warning('Ingresa tu usuario y password')

