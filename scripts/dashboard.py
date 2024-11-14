import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# player information (current globals)
df = pd.read_csv(f'../data/players.csv')
players = df.name.values.tolist()
df_teams = df.groupby('team').sum()['total_points'].reset_index().sort_values('total_points', ascending=False)


st.set_page_config(
    page_title="Fantasy Premier League",
    layout="wide",
    initial_sidebar_state="expanded")

class Dashboard(object):


    def __init__(self):

        self.columns = (5, 1.5)
        self.col = st.columns(self.columns, gap='medium')

    # def configure(self):
    #     ##### main page config
        

    def create_field(self):

        with self.col[0]:

            # Set up the Streamlit UI
            st.title("Team Field Positioning")

            img = mpimg.imread("../data/field.png")  # Assume you have an image of the field in the same directory

            # position = player_positions[selected_player]
            # st.write(f"Position on field: {position}")

            # Plot the field and player
            fig, ax = plt.subplots(figsize=(3,3))

            # Display the field image
            ax.imshow(img)

            # Hide axes for a cleaner look
            ax.set_xticks([])
            ax.set_yticks([])

            # Display the plot
            st.pyplot(fig, use_container_width=False)


    def create_topteams(self):


        with self.col[1]:
            # st.markdown('#### Top States')

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
    dash.create_field()
    dash.create_topteams()

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