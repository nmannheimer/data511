# team_computation.py

import pandas as pd
from constants import FORMATION_MAP
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import umap

def get_top_players_by_position(player_data, formation):
    """
    Selects the top players by position based on the following priority:
    1. Players in the Dream Team (`in_dreamteam` is True).
    2. Higher Dream Team Count (`dreamteam_count`).
    3. Total Points (`total_points`).

    Parameters:
    - player_data (pd.DataFrame): DataFrame containing player statistics.
    - formation (str): Selected team formation (e.g., '4-4-2').

    Returns:
    - best_team (list of dict): List of selected player dictionaries for the best team.
    """
    position_counts = FORMATION_MAP[formation]
    best_team = []

    for position, count in position_counts.items():
        # Filter players by position
        position_players = player_data[player_data['position'] == position]

        # Ensure necessary columns exist
        required_columns = ['in_dreamteam', 'dreamteam_count', 'total_points']
        for col in required_columns:
            if col not in position_players.columns:
                position_players[col] = 0  # Assign default value if column is missing

        # Sort players by 'in_dreamteam', 'dreamteam_count', then 'total_points' in descending order
        top_players = position_players.sort_values(
            by=['in_dreamteam', 'dreamteam_count', 'total_points'],
            ascending=[False, False, False]
        )

        # Select the top 'count' players
        selected_players = top_players.head(count).to_dict('records')
        best_team.extend(selected_players)

    return best_team

def adjust_team_to_budget(team, budget, player_data):
    """
    Adjusts the team to fit within the budget by replacing expensive players with cheaper alternatives.

    Parameters:
    - team (list of dict): The current team list.
    - budget (int): The total budget in tenths of millions.
    - player_data (pd.DataFrame): DataFrame containing player statistics.

    Returns:
    - adjusted_team (list of dict): The adjusted team list within budget constraints.
    """
    total_cost = sum(player['now_cost'] for player in team)
    if total_cost <= budget:
        return team  # Team is within budget

    # Create a copy of the team to avoid modifying the original list
    adjusted_team = team.copy()
    # Sort the team by cost in descending order
    adjusted_team.sort(key=lambda x: x['now_cost'], reverse=True)

    # Iterate over players and replace with cheaper alternatives
    for i in range(len(adjusted_team)):
        if total_cost <= budget:
            break  # Team is now within budget

        expensive_player = adjusted_team[i]
        position = expensive_player['position']

        # Find a cheaper alternative
        cheaper_players = player_data[
            (player_data['position'] == position) &
            (~player_data['web_name'].isin([p['web_name'] for p in adjusted_team])) &
            (player_data['now_cost'] < expensive_player['now_cost'])
        ].sort_values(by='total_points', ascending=False)

        # Replace with the next best cheaper player
        if not cheaper_players.empty:
            candidate = cheaper_players.iloc[0]
            adjusted_team[i] = candidate.to_dict()
            total_cost = sum(player['now_cost'] for player in adjusted_team)
        else:
            continue  # No cheaper players available for this position

    return adjusted_team

def get_similar_players(df: pd.DataFrame, player_name:str, top_n: int = 5):
    
    """Returns top_n similar players based on the player input. Uses UMAP for dimensional reduction and euclidean distance to find similarities
       Requires sklearn, scipy and umap modules.   
    """
    
    df["now_cost_m"] = df["now_cost"]/10
    target_position = df.loc[df['web_name'] == player_name, 'position'].values[0]
    if target_position == 'GKP':
        numeric_features = ["now_cost_m", "total_points", "minutes", "goals_conceded", "clean_sheets", "ict_index"]
    elif target_position == 'DEF':
        numeric_features = ["now_cost_m", "total_points", "minutes", "goals_conceded", "clean_sheets", "assists", "creativity", "goals_scored", "ict_index"]
    elif target_position == "MID":
        numeric_features = ["now_cost_m", "total_points", "minutes", "goals_scored", "assists", "creativity", "influence", "threat", "goals_conceded", "clean_sheets", "ict_index"]
    else:
        numeric_features = ["now_cost_m", "total_points", "minutes", "goals_scored", "assists", "creativity", "influence", "threat", "ict_index"]
    metadata = ['web_name', 'position']
    filtered_data = df[metadata + numeric_features].dropna()

    
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(filtered_data[numeric_features])

    umap_reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2)
    umap_features = umap_reducer.fit_transform(normalized_features)

    distance_matrix_umap = cdist(umap_features, umap_features, metric='euclidean')

    distance_df_umap = pd.DataFrame(distance_matrix_umap, index=filtered_data['web_name'], columns=filtered_data['web_name'])

    if player_name not in distance_df_umap.index:
        return f"Player '{player_name}' not found in the dataset."

    same_position_players = df[df['position'] == target_position]['web_name']
    distances = distance_df_umap.loc[player_name, same_position_players]
    similar_players = distances.sort_values()[1:top_n+1]  # Exclude self (distance = 0)
    return similar_players