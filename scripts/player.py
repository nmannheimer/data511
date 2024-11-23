import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from copy import copy

from streamlit_elements import elements, mui, html, nivo
from PIL import Image
import requests
from io import BytesIO

def get_prof_pic(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        print(f"Failed to fetch image. HTTP Status Code: {response.status_code}")
        return np.zeros((110,140,3))

# Load and preprocess data
# @st.cache_data
# def load_player_data_from_api():
try:
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    st.error(f"There was an error: {e} while retrieving data")
    # return pd.DataFrame()  # Return empty DataFrame on error

# Load data into DataFrames
df_elements = pd.DataFrame(data["elements"])
df_element_types = pd.DataFrame(data["element_types"])
df_teams = pd.DataFrame(data["teams"])

# Merge df_elements and df_element_types on 'element_type' and 'id'
df_merged = df_elements.merge(
    df_element_types[['id', 'plural_name_short', 'plural_name']],
    left_on='element_type',
    right_on='id',
    how='left'
)

# Merge with teams to get team names
df_merged = df_merged.merge(
    df_teams[['id', 'name']],
    left_on='team',
    right_on='id',
    how='left',
    suffixes=('', '_team')
)

# Rename columns
df_merged.rename(columns={
    'plural_name_short': 'position',
    'name': 'team_name',
    'photo': 'photo',
    'code': 'code'
}, inplace=True)

# Drop redundant columns
df_merged.drop(columns=['id_y', 'id'], inplace=True)
df_merged.rename(columns={'id_x': 'id'}, inplace=True)

# Standardize 'position' to upper case
df_merged['position'] = df_merged['position'].str.upper()

# Add 'player_value_score' (custom metric, here set equal to 'total_points')
df_merged['player_value_score'] = df_merged['total_points']

# Construct player photo URLs
df_merged['photo_url'] = df_merged.apply(
    lambda row: f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{row['code']}.png",
    axis=1
)

# Select the desired columns
columns_to_use = [
    'id', 'web_name', 'first_name', 'second_name', 'position', 'plural_name', 'now_cost',
    'total_points', 'player_value_score', 'minutes', 'goals_scored',
    'assists', 'clean_sheets', 'goals_conceded', 'yellow_cards',
    'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity',
    'threat', 'ict_index', 'selected_by_percent', 'form', 'points_per_game',
    'team_name', 'in_dreamteam', 'dreamteam_count', 'photo_url'
]

# Ensure all selected columns exist in the DataFrame
df = df_merged[columns_to_use]

df['full_name'] = df['first_name'] + ' ' + df['second_name']

############################

# players = df.name.values.tolist()
players = df.full_name.values.tolist()

# df_teams = df.groupby('team').sum()['total_points'].reset_index().sort_values('total_points', ascending=False)
# pie_data=[]
# for i in range(12):
    # pie_data.append({"id": df.iloc[i]['name'], "label": df.iloc[i]['name'], "value": df.iloc[i]['now_cost']/10.})

BUDGET = 100
############################################

# st.set_page_config(
#     page_title="Fantasy Premier League",
#     layout="wide",
#     initial_sidebar_state="expanded")

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





##############################
dash = Dashboard()
dash.set_columns((3,5,3))
# dash.create_field(col_num=0)
# dash.create_topteams(col_num=1)

# timeline = st.sidebar.slider("Timeline", 0.0, 1.0, 0.5)

# Sidebar for player selection
# with st.sidebar.expander("Team Selection", expanded=True):
    # selected_players = st.sidebar.multiselect("Select a player", players)

if 'selected_player0' in st.session_state:
    selected_player0 = st.session_state.selected_player0
    selected_player0_idx = df.full_name.tolist().index(selected_player0)
else:
    selected_player0 = None
    selected_player0_idx = 0

if 'selected_player1' in st.session_state:
    selected_player1 = st.session_state.selected_player1
    selected_player1_idx = df.full_name.tolist().index(selected_player1)
else:
    selected_player1 = None
    selected_player1_idx = 0

with dash.col[0]:
    selected_player0 = st.selectbox(
        "Select a player 1:",
        players,
        index=selected_player0_idx
    )

st.session_state.selected_player0 = selected_player0

with dash.col[2]:
    selected_player1 = st.selectbox(
        "Select a player 2:",
        players,
        index=selected_player1_idx
    )

st.session_state.selected_player1 = selected_player1
selected_players = [st.session_state.selected_player0, st.session_state.selected_player1]


############################## create demo charts
pie_data=[]
remaining_budget = copy(BUDGET)
total_cost = 0
metrics = ['creativity', 'threat','ict_index','selected_by_percent','form']

radar_data = [{"metric": metrics[0]},
                {"metric": metrics[1]},
                {"metric": metrics[2]},
                {"metric": metrics[3]},
                {"metric": metrics[4]}
            ]           
keys_ = [m['metric'] for m in radar_data]
for p in selected_players:
    current_cost = df[df.full_name==p].now_cost / 10.
    cost_perc = 100. * (current_cost.iloc[0] / BUDGET)
    pie_data.append({"id": p, "label": p, "value": f'{cost_perc:0.2f}'})

    player_name = df[df.full_name==p]['full_name'].iloc[0]
    radar_data[0][player_name] = float(df[df.full_name==p][metrics[0]].iloc[0])
    radar_data[2][player_name] = float(df[df.full_name==p][metrics[1]].iloc[0])
    radar_data[1][player_name] = float(df[df.full_name==p][metrics[2]].iloc[0])
    radar_data[3][player_name] = float(df[df.full_name==p][metrics[3]].iloc[0])
    radar_data[4][player_name] = float(df[df.full_name==p][metrics[4]].iloc[0])

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
    st.markdown(f'### {st.session_state.selected_player0}', unsafe_allow_html=True)
    url0 = df[df.full_name==st.session_state.selected_player0].photo_url.values[0]
    pic0 = get_prof_pic(url0)
    pic0 = np.array(pic0)
    pic0[pic0.sum(-1) == 255*3] = 0 #flip background color to match dark theme
    st.image(pic0)


with dash.col[2]:
    st.markdown(f'### {st.session_state.selected_player1}', unsafe_allow_html=True)
    url1 = df[df.full_name==st.session_state.selected_player1].photo_url.values[0]
    pic1 = get_prof_pic(url1)
    pic1 = np.array(pic1)
    pic1[pic1.sum(-1) == 255*3] = 0
    st.image(pic1)

with dash.col[1]:
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