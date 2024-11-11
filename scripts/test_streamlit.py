# import streamlit as st
# import wikipedia
# from streamlit_searchbox import st_searchbox
import pandas as pd


import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


df = pd.read_csv(f'../data/players.csv')
players = df.name.values.tolist()

# def search_list(searchterm: str) -> list:
#     return [players.index(searchterm)] if searchterm else []

# # pass search function and other options as needed
# selected_value = st_searchbox(
#     search_list,
#     placeholder="Search Players... ",
#     key="my_key",
# )

# st.write(f"Selected value: {selected_value}")

# Set up the Streamlit UI
st.title("Team Field Positioning")

col1, col2 = st.columns(2)

# Sidebar for player selection
with  col2:
    selected_player = st.sidebar.selectbox("Select a player", players)

# Load the field image
with col1:
    img = mpimg.imread("../data/field.png")  # Assume you have an image of the field in the same directory

# Display the selected player and their position
    st.write(f"Selected player: {selected_player}")
# position = player_positions[selected_player]
# st.write(f"Position on field: {position}")

# Plot the field and player
fig, ax = plt.subplots(figsize=(3,3))

# Display the field image
ax.imshow(img)

# Plot the player's position on the field
# ax.plot(position[1], position[0], 'ro', markersize=10)  # 'ro' for red dot

# Add text to the position
# ax.text(position[1] + 0.1, position[0], selected_player, color='white', fontsize=12, ha='left')

# Hide axes for a cleaner look
ax.set_xticks([])
ax.set_yticks([])

# Display the plot
st.pyplot(fig)

