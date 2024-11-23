import streamlit as st

st.set_page_config(
    page_title="Fantasy Premier League",
    layout="wide",
    initial_sidebar_state="expanded")

pg = st.navigation([st.Page("dash.py"), st.Page("test_nivo.py")])
pg.run()