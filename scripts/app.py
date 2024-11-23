import streamlit as st

st.set_page_config(
    page_title="Fantasy Premier League",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation([st.Page("team.py", title='Team Selection'), st.Page("player.py", title='Player Comparison')])
pg.run()