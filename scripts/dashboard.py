import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from streamlit_elements import elements, mui, html, nivo


# player information (current globals)
df = pd.read_csv(f'../data/players.csv')
players = df.name.values.tolist()
df_teams = df.groupby('team').sum()['total_points'].reset_index().sort_values('total_points', ascending=False)
pie_data=[]
for i in range(12):
    pie_data.append({"id": df.iloc[i]['name'], "label": df.iloc[i]['name'], "value": df.iloc[i]['now_cost']/10.})
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


    timeline = st.sidebar.slider("Timeline", 0.0, 1.0, 0.5)

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


    with st.sidebar.expander('Budget', expanded=True):

        with elements("nivo_pie_chart"):
            with mui.Box(sx={"height": 500}):
                nivo.Pie(
                    data=pie_data,
                    margin={"top": 40, "right": 80, "bottom": 80, "left": 80},
                    innerRadius=0.5,
                    padAngle=0.7,
                    cornerRadius=3,
                    colors={ "scheme": "nivo" },
                    borderWidth=1,
                    borderColor={"from": "color", "modifiers": [["darker", 0.2]]},
                    radialLabelsSkipAngle=10,
                    radialLabelsTextColor="#333333",
                    radialLabelsLinkColor={"from": "color"},
                    sliceLabelsSkipAngle=10,
                    sliceLabelsTextColor="#333333"
                )