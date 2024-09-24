import streamlit as st
from streamlit import session_state as ss
import process.session as Session

if 'authentication_status' not in ss or not ss['authentication_status']:
    st.switch_page('./pages/account.py')


col1, col2, col3 = st.columns(3)
with col2:
    st.image('assets/dr-igualin.png', width=200)
    st.header('Dr. Igualin')

st.markdown("Terminos y condiciones de uso")


state = st.checkbox("Acepto los [ToS](https://docs.google.com/document/d/1CGBiMT5vR90Gb45vfMpcmEytfTQberkpdNYuygsyEBg/edit) y la [política de privacidad](https://docs.google.com/document/d/18-0edIOx3m2TJiDwMCIdeNv2cyk28LrEO7bYMuvD7vc/edit#heading=h.d6iqeqvv2yt6)")

if st.button("Continuar"):
    if state:
        Session.accept_tos()
        st.switch_page('./chat.py')
    else:
        st.error("Debes aceptar los Tos")
