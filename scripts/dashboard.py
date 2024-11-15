import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from copy import copy

from streamlit_elements import elements, mui, html, nivo

# player information (currently globals)
df = pd.read_csv(f'../data/players.csv')
players = df.name.values.tolist()
df_teams = df.groupby('team').sum()['total_points'].reset_index().sort_values('total_points', ascending=False)
pie_data=[]
for i in range(12):
    pie_data.append({"id": df.iloc[i]['name'], "label": df.iloc[i]['name'], "value": df.iloc[i]['now_cost']/10.})

BUDGET = 100 #USD?
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
            # st.title("Team Field Positioning")
            img = mpimg.imread("../data/dark_field.png")  # Assume you have an image of the field in the same directory
            fig, ax = plt.subplots(figsize=(3,3), frameon=False)
            ax.imshow(img)
            plt.tight_layout()
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig, use_container_width=False)

            # fig, ax = plt.subplots(figsize=(3,3))
            # matplotsoccer.field("green",figsize=8, show=False)
            # plt.scatter(x,y)
            # plt.axis("on")
            # plt.show()
            # st.pyplot(fig, use_container_width=False)

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
    dash.set_columns((5,5))
    # dash.create_field(col_num=0)
    # dash.create_topteams(col_num=1)

    # timeline = st.sidebar.slider("Timeline", 0.0, 1.0, 0.5)

    # Sidebar for player selection
    with st.sidebar.expander("Team Selection", expanded=False):
        selected_players = st.sidebar.multiselect("Select a player", players)

        pie_data=[]
        remaining_budget = copy(BUDGET)
        total_cost = 0
        radar_data = [{"metric": 'points_per_game_rank'},
                      {"metric": 'influence_rank'},
                      {"metric": 'now_cost_rank'},
                      {"metric": 'selected_rank'},
                      {"metric": 'ict_index_rank_type'}
                    ]           
        keys_ = [m['metric'] for m in radar_data]
        for p in selected_players:
            current_cost = df[df.name==p].now_cost / 10.
            cost_perc = 100. * (current_cost.iloc[0] / BUDGET)
            pie_data.append({"id": p, "label": p, "value": f'{cost_perc:0.2f}'})

            player_name = df[df.name==p]['name'].iloc[0]
            radar_data[0][player_name] = int(df[df.name==p]['points_per_game_rank'].iloc[0])
            radar_data[2][player_name] = int(df[df.name==p]['influence_rank'].iloc[0])
            radar_data[1][player_name] = int(df[df.name==p]['now_cost_rank'].iloc[0])
            radar_data[3][player_name] = int(df[df.name==p]['selected_rank'].iloc[0])
            radar_data[4][player_name] = int(df[df.name==p]['ict_index_rank_type'].iloc[0])

    # st.write(f"Selected player: {selected_players}")
    # with st.sidebar.expander("Team Performance", expanded=True):
    #     # TODO: replace this with actual data....
    #     x = np.linspace(0, 10, 100)
    #     y = np.sin(x)
    #     fig, ax = plt.subplots()
    #     ax.plot(x, y)
    #     ax.set(xlabel='X-axis', ylabel='Y-axis', title='REPLACE')
    #     st.pyplot(fig)

    # with st.sidebar.expander('Current Team Selection', expanded=True):

    with dash.col[0]:
        with elements("nivo_pie_chart"):
            with mui.Box(sx={"height": 300}):
                nivo.Pie(
                    data=pie_data,
                    margin={"top": 30, "right": 80, "bottom": 50, "left": 80},
                    innerRadius=0.5,
                    padAngle=0.7,
                    cornerRadius=3,
                    colors={ "scheme": "nivo" },
                    borderWidth=1,
                    borderColor={"from": "color", "modifiers": [["darker", 0.2]]},
                    radialLabelsSkipAngle=10,
                    radialLabelsTextColor="black",
                    radialLabelsLinkColor={"from": "color"},
                    sliceLabelsSkipAngle=10,
                    sliceLabelsTextColor="#ffffff",
                    arcLabelsTextColor='#ffffff',
                    arcLinkLabelsTextColor='#ffffff',
                    arcLinkLabelsColor='#ffffff'
                )

    with dash.col[1]:
        with elements("nivo_charts"):
            with mui.Box(sx={"height": 500}):
                nivo.Radar(
                    data=radar_data,
                    keys=selected_players,
                    indexBy="metric",
                    valueFormat=">-.2f",
                    margin={ "top": 70, "right": 80, "bottom": 40, "left": 80 },
                    # borderColor={ "from": "color" },
                    gridLabelOffset=36,
                    dotSize=10,
                    # dotColor={ "theme": "background" },
                    dotBorderWidth=2,
                    motionConfig="wobbly",
                    legends=[
                        {
                            "anchor": "top-left",
                            "direction": "column",
                            "translateX": -50,
                            "translateY": -40,
                            "itemWidth": 80,
                            "itemHeight": 20,
                            # "itemTextColor": "#ffffff",
                            "symbolSize": 12,
                            "symbolShape": "circle",
                            "effects": [
                                {
                                    "on": "hover",
                                    # "style": {
                                        # "itemTextColor": "#ffffff"
                                    # }
                                }
                            ]
                        }
                    ],
                    theme={
                        # "background": "#FFFFFF",
                        "textColor": "#ffffff",
                        "tooltip": {
                            # "container": {
                                # "background": "#FFFFFF",
                                # "color": "#ffffff",
                            # }
                        }
                    }
                )