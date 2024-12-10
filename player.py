import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from copy import copy
from PIL import Image
import requests
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import umap

# repo
from utils.data_loader import load_player_data_from_api, load_gameweek_data_from_github
from visualizations import plot_transfers_in_out_by_player, plot_fpl_performance_funnel, plot_gw_performance_by_player, radar_chart_player_comparison

def format_keys(metrics):
    formatted_keys =  [' '.join(word.capitalize() for word in metric.split('_')) for metric in metrics]
    # formatted_keys = [k.replace(' ', '\n') for k in formatted_keys]
    return formatted_keys

def get_prof_pic(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        print(f"Failed to fetch image. HTTP Status Code: {response.status_code}")
        return Image.fromarray(np.zeros((110,140,3), dtype=np.uint8))

@st.cache_data()
def get_similar_players(df_new: pd.DataFrame, player_name:str, target_position=None, top_n: int = 5):
    
    """Returns top_n similar players based on the player input. Uses UMAP for dimensional reduction and euclidean distance to find similarities
       Requires sklearn, scipy and umap modules.   
    """
    df = df_new.copy()

    df["now_cost_m"] = df["now_cost"]/10
    
    # target_position = df[df.web_name == player_name].position.values[0]
    # df[df.web_name == 'Havertz'].position.values[0]

    if target_position == 'GKP':
        numeric_features = ["now_cost_m", "total_points", "minutes", "goals_conceded", "clean_sheets", "ict_index"]
    elif target_position == 'DEF':
        numeric_features = ["now_cost_m", "total_points", "minutes", "goals_conceded", "clean_sheets", "assists", "creativity", "goals_scored", "ict_index"]
    elif target_position == "MID":
        numeric_features = ["now_cost_m", "total_points", "minutes", "goals_scored", "assists", "creativity", "influence", "threat", "goals_conceded", "clean_sheets", "ict_index"]
    else:
        numeric_features = ["now_cost_m", "total_points", "minutes", "goals_scored", "assists", "creativity", "influence", "threat", "ict_index"]
    metadata = ['full_name', 'position']
    filtered_data = df[metadata + numeric_features].dropna()
    
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(filtered_data[numeric_features])

    umap_reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, random_state=1337)
    umap_features = umap_reducer.fit_transform(normalized_features)

    distance_matrix_umap = cdist(umap_features, umap_features, metric='euclidean')

    distance_df_umap = pd.DataFrame(distance_matrix_umap, index=filtered_data['full_name'], columns=filtered_data['full_name'])

    if player_name not in distance_df_umap.index:
        return f"Player '{player_name}' not found in the dataset."

    same_position_players = df[df['position'] == target_position]['full_name']
    distances = distance_df_umap.loc[player_name, same_position_players]

    # Normalize similarity scores (invert distances and normalize)
    similarity_scores = 1 / (1 + distances)  # Invert distances to get similarity
    normalized_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min())

    # Get top N most similar players (excluding the player itself)
    similar_players = normalized_scores.sort_values(ascending=False)[1:top_n + 1]  # Exclude self (similarity = 1)

    return similar_players

############################
df = load_player_data_from_api()

year = '2024-25'
df_gh = load_gameweek_data_from_github(year)

# players = df.name.values.tolist()
players = sorted(df.full_name.values.tolist())

# df_teams = df.groupby('team').sum()['total_points'].reset_index().sort_values('total_points', ascending=False)
# pie_data=[]
# for i in range(12):
    # pie_data.append({"id": df.iloc[i]['name'], "label": df.iloc[i]['name'], "value": df.iloc[i]['now_cost']/10.})

BUDGET = 100
############################################

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
dash.set_columns((4,4,4))

for key, val in st.session_state.items():
    st.session_state[key] = val

with dash.col[0]:
    player0 = st.selectbox(
        "Select first player:",
        players,
        index=None, #default value for user not come to an empty page, required
        key='p0'
    )

with dash.col[2]:
    player1 = st.selectbox(
        "Select second player:",
        players,
        index=None, #default value for user not come to an empty page, not required
        key='p1'
    )

if (player0 is not None and player1 is None) or (player0 is None and player1 is not None):
    player = copy(player0 if player0 is not None else player1)
    player_position = str(df[df.full_name==player].position.values[0])
    sim_players = get_similar_players(df, player, target_position=player_position, top_n=5)
    
    print(sim_players)

    with dash.col[1]:
        sim_players_df = pd.DataFrame(sim_players)
        sim_players_df = sim_players_df.reset_index()
        sim_players_df.rename(columns={"full_name": "Similar Players", player: 'Similarity Score'}, inplace=True)
        sim_players_df.set_index('Similar Players', inplace=True)
        st.write(sim_players_df)

if player0 is not None and player1 is not None:

    #update states
    st.session_state.selected_player0 = player0
    st.session_state.selected_player1 = player1
    selected_players = [st.session_state.selected_player0, st.session_state.selected_player1]

    ############################## create demo charts
    pie_data=[]
    remaining_budget = copy(BUDGET)
    total_cost = 0
    metrics = ['now_cost', 'total_points','goals_conceded','creativity','form']

    metrics_formatted = format_keys(metrics)

    # radar_data = [{"metric": metrics_formatted[0]},
    #               {"metric": metrics_formatted[1]},
    #               {"metric": metrics_formatted[2]},
    #               {"metric": metrics_formatted[3]},
    #               {"metric": metrics_formatted[4]}
    #             ]
    # keys_ = [m['metric'] for m in radar_data]
    # for p in selected_players:
    #     current_cost = df[df.full_name==p].now_cost / 10.
    #     cost_perc = 100. * (current_cost.iloc[0] / BUDGET)
    #     pie_data.append({"id": p, "label": p, "value": f'{cost_perc:0.2f}'})

    #     radar_data[0][p] = float(df[df.full_name==p][metrics[0]].iloc[0]) if df[df.full_name==p][metrics[0]].iloc[0] != '' else 0
    #     radar_data[2][p] = float(df[df.full_name==p][metrics[1]].iloc[0]) if df[df.full_name==p][metrics[1]].iloc[0] != '' else 0
    #     radar_data[1][p] = float(df[df.full_name==p][metrics[2]].iloc[0]) if df[df.full_name==p][metrics[2]].iloc[0] != '' else 0
    #     radar_data[3][p] = float(df[df.full_name==p][metrics[3]].iloc[0]) if df[df.full_name==p][metrics[3]].iloc[0] != '' else 0
    #     radar_data[4][p] = float(df[df.full_name==p][metrics[4]].iloc[0]) if df[df.full_name==p][metrics[4]].iloc[0] != '' else 0

    with dash.col[0]:
        st.markdown(f'#### {st.session_state.selected_player0}', unsafe_allow_html=True)
        url0 = df[df.full_name==st.session_state.selected_player0].photo_url.values[0]
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="{url0}" style="width:200px; border-radius:10%;">
            </div>
            """,
            unsafe_allow_html=True
        )

        # transfers plot
        plot_transfers_in_out_by_player(player0, df_gh)

        st.divider()

        plot_gw_performance_by_player(player0, df_gh)

    with dash.col[2]:
        st.markdown(f'#### {st.session_state.selected_player1}', unsafe_allow_html=True)
        url1 = df[df.full_name==st.session_state.selected_player1].photo_url.values[0]
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="{url1}" style="width:200px; border-radius:10%;">
            </div>
            """,
            unsafe_allow_html=True
        )

        # transfers plot
        plot_transfers_in_out_by_player(player1, df_gh)

        st.divider()

        plot_gw_performance_by_player(player1, df_gh)

    with dash.col[1]:
        radar_chart_player_comparison(df, player0, player1, 
                                       metrics = ['total_points', 'minutes', 'goals_scored', 
                                                  'assists', 'goals_conceded', 'clean_sheets', 'selected_by_percent'])
        st.divider()

    with dash.col[1]:
        # julian's plot
        plot_fpl_performance_funnel(df_gh, [player0, player1], player='name',
                                total_points_column='total_points', xp_column='xP')
        