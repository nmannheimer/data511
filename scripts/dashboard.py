import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# player information (current globals)
df = pd.read_csv(f'../data/players.csv')
players = df.name.values.tolist()
df_teams = df.groupby('team').sum()['total_points'].reset_index().sort_values('total_points', ascending=False)
############################################

st.set_page_config(
    page_title="Fantasy Premier League",
    layout="wide",
    initial_sidebar_state="expanded")

class Dashboard(object):

    def __init__(self):
        self.columns = None

    def set_columns(self, cols):
        self.col = st.columns(cols, gap='medium')

    def create_field(self, col_num):

        with self.col[col_num]:
            st.title("Team Field Positioning")
            img = mpimg.imread("../data/field.png")  # Assume you have an image of the field in the same directory
            fig, ax = plt.subplots(figsize=(3,3))
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig, use_container_width=False)

    def create_topteams(self, col_num):

        with self.col[col_num]:
            st.dataframe(df_teams,
                        column_order=("team", "total_points"),
                        hide_index=True,
                        width=None,
                        height=500,
                        column_config={
                            "team": st.column_config.TextColumn(
                                "Team",
                            ),
                            "total_points": st.column_config.ProgressColumn(
                                "Total Points",
                                format="%f",
                                min_value=0,
                                max_value=max(df_teams.total_points),
                            )}
                        )


if __name__ == '__main__':

    dash = Dashboard()
    dash.set_columns((3,2))
    dash.create_field(col_num=0)
    dash.create_topteams(col_num=1)

    # Sidebar for player selection
    with st.sidebar.expander("Team Selection", expanded=False):
        selected_player = st.sidebar.selectbox("Select a player", players)

    st.write(f"Selected player: {selected_player}")

    with st.sidebar.expander("Team Performance", expanded=True):
        # TODO: replace this with actual data....
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='X-axis', ylabel='Y-axis', title='REPLACE')
        st.pyplot(fig)