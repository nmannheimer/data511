import os

# Define the content to write
config_content = """
[theme]
base = "dark"
"""

# Get the path to the home directory
home_dir = os.path.expanduser("~")

# Define the filename
os.makedirs(os.path.join(home_dir, ".streamlit/"), mode=0o777, exist_ok=True)
file_path = os.path.join(home_dir, ".streamlit/config.toml")

# Write the content to the file
with open(file_path, "w") as file:
    file.write(config_content)


#########

import streamlit as st

for k, v in st.session_state.items():
    st.session_state[k] = v

st.set_page_config(
    page_title="Fantasy Premier League",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",

)

pg = st.navigation([st.Page("team.py", title='Team Selection'), st.Page("player.py", title='Player Comparison')])
pg.run()