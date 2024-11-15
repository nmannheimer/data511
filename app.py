import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title for the app
st.title("Fantasy Premier League")

# Read player data from the CSV file
df = pd.read_csv("players.csv")


# Function to calculate player value score based on position-specific factors
def calculate_player_value(data):
    if data['position'] == 'GKP':
        return data['bonus'] + data['clean_sheets'] * 2.5 + data['penalties_saved'] * 2 + data['saves'] * 1.5 - data[
            'expected_goals_conceded'] * 0.5
    elif data['position'] == 'DEF':
        return data['bonus'] + data['expected_goals_conceded'] * -1 + data['clean_sheets'] * 2 + data['threat'] + data[
            'influence'] * 0.5
    elif data['position'] == 'MID':
        return data['expected_goals_per_90'] * 2 + data['expected_assists'] + data['creativity'] + data['influence']
    elif data['position'] == 'FWD':
        return data['bonus'] + data['expected_goals_per_90'] * 2 + data['goals_scored'] + data['threat'] * 0.5
    return 0


# Apply the value calculation function to each player
df['player_value_score'] = df.apply(calculate_player_value, axis=1)

# Sidebar for team selection options
st.sidebar.title("Team Selection")
formation = st.sidebar.selectbox("Select Formation", ['3-4-3', '3-5-2', '4-4-2', '4-3-3'])
budget = 1000000  # = st.sidebar.slider("Budget", min_value=int(df['now_cost'].min()), max_value=1000, value=100)

# Define formations with positions and required players
formation_map = {
    '3-4-3': {'GKP': 1, 'DEF': 3, 'MID': 4, 'FWD': 3},
    '3-5-2': {'GKP': 1, 'DEF': 3, 'MID': 5, 'FWD': 2},
    '4-4-2': {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2},
    '4-3-3': {'GKP': 1, 'DEF': 4, 'MID': 3, 'FWD': 3},
}

# Display formation and set player counts based on the selection
st.sidebar.write("Formation:", formation)
position_counts = formation_map[formation]

selected_players = []
total_cost = 0


def limit_position_selection(position, count, remaining_budget):
    """
    Function to limit the selection of players for a given position based on the remaining budget.

    Parameters:
    position (str): The position of the players (e.g., 'GKP', 'DEF', 'MID', 'FWD').
    count (int): The number of players to select for the given position.
    remaining_budget (float): The remaining budget for selecting players.

    Returns:
    list: A list of selected players for the given position.
    """
    available_players = df[df['position'] == position].sort_values(by='player_value_score', ascending=False)
    selected = st.sidebar.multiselect(
        f"Select {count} {position} players",
        options=available_players['name'].tolist(),
        max_selections=count,
        default=[],
        help=f"Select up to {count} {position} players within budget."
    )

    position_players = []
    for player_name in selected:
        player = available_players[available_players['name'] == player_name].iloc[0]
        if total_cost + player['now_cost'] <= budget:
            position_players.append({
                'Position': position,
                'Name': player['name'],
                'Cost': player['now_cost'],
                'Value': player['player_value_score']
            })
            total_cost += player['now_cost']

    return position_players


# Loop through each position and select players
for position, count in position_counts.items():
    # Filter and sort players by position and value score
    position_selected_players = df[df['position'] == position].sort_values(by='player_value_score', ascending=False)

    # Filter players within remaining budget
    remaining_budget_per_player = (budget - total_cost) / count
    available_players = position_selected_players[position_selected_players['now_cost'] <= remaining_budget_per_player]

    # Multi-select dropdown for player selection within budget
    selected = st.sidebar.multiselect(
        f"Select {count} {position} players",
        options=available_players['name'].tolist(),
        default=[],
        help=f"Select up to {count} {position} players within budget."
    )

    # Add selected players to the team list if within budget
    for player_name in selected:
        player = available_players[available_players['name'] == player_name].iloc[0]
        if total_cost + player['now_cost'] <= budget:
            selected_players.append({
                'Position': position,
                'Name': player['name'],
                'Cost': player['now_cost'],
                'Value': player['player_value_score']
            })
            total_cost += player['now_cost']

if selected_players:
    st.write("Selected Team")
    st.write(f"Total Cost: {total_cost} / {budget}")
    team_df = pd.DataFrame(selected_players)
    st.dataframe(team_df[['Position', 'Name', 'Cost', 'Value']].style.background_gradient(cmap='viridis'))
else:
    st.write("No players selected.")

# Display a warning if budget exceeded
if total_cost > budget:
    st.warning("Budget exceeded. Please adjust your selections.")


def draw_soccer_field(ax):
    """
    Draws a soccer field on the given axes.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to draw the soccer field.
    """
    # Draw the field boundaries
    ax.plot([0, 0, 90, 90, 0], [0, 65, 65, 0, 0], color="green")
    # Draw the center line
    ax.plot([45, 45], [0, 65], color="green")
    # Draw the penalty areas
    ax.plot([0, 16.5, 16.5, 0], [13, 13, 52, 52], color="green")
    ax.plot([90, 73.5, 73.5, 90], [13, 13, 52, 52], color="green")
    # Draw the goal areas
    ax.plot([0, 5.5, 5.5, 0], [24.5, 24.5, 40.5, 40.5], color="green")
    ax.plot([90, 84.5, 84.5, 90], [24.5, 24.5, 40.5, 40.5], color="green")
    # Draw the center circle
    center_circle = plt.Circle((45, 32.5), 9.15, color="green", fill=False)
    ax.add_patch(center_circle)
    # Draw the center spot
    ax.plot(45, 32.5, 'o', color="green")
    # Set the limits and aspect
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 65)
    ax.set_aspect(1)
    ax.axis('off')


def get_formation_map():
    """
    Returns a dictionary mapping formations to player positions and their coordinates.

    Returns:
    dict: A dictionary where keys are formation names and values are dictionaries
          mapping player positions to their coordinates on the field.
    """
    return {
        '3-4-3': {'GKP': [(5, 32.5)], 'DEF': [(20, 15), (20, 32.5), (20, 50)],
                  'MID': [(40, 10), (40, 25), (40, 40), (40, 55)],
                  'FWD': [(70, 20), (70, 32.5), (70, 45)]},
        '3-5-2': {'GKP': [(5, 32.5)], 'DEF': [(20, 15), (20, 32.5), (20, 50)],
                  'MID': [(40, 5), (40, 20), (40, 32.5), (40, 45), (40, 60)],
                  'FWD': [(70, 25), (70, 40)]},
        '4-4-2': {'GKP': [(5, 32.5)], 'DEF': [(20, 10), (20, 25), (20, 40), (20, 55)],
                  'MID': [(40, 15), (40, 30), (40, 45), (40, 55)],
                  'FWD': [(70, 25), (70, 40)]},
        '4-3-3': {'GKP': [(5, 32.5)], 'DEF': [(20, 10), (20, 25), (20, 40), (20, 55)],
                  'MID': [(40, 20), (40, 32.5), (40, 45)],
                  'FWD': [(70, 15), (70, 32.5), (70, 50)]},
    }


# Function to plot players on the field
def plot_players_on_field(selected_team, formation):
    """
    Plots the players on the soccer field based on the selected team and formation.

    Parameters:
    selected_team (list): A list of dictionaries containing player information.
    formation (str): The formation to be used for plotting players.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    draw_soccer_field(ax)

    formation_map = get_formation_map()
    player_positions = formation_map.get(formation, {})

    for position, coordinates in player_positions.items():
        position_players = [p for p in selected_team if p['Position'] == position]
        for i, (x, y) in enumerate(coordinates):
            if i < len(position_players):
                player = position_players[i]
                ax.text(x, y, player['Name'], ha='center', color="blue", fontsize=12, fontweight="bold",
                        bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
                ax.plot(x, y, 'o', color="blue", markersize=10)
            else:
                ax.plot(x, y, 'o', color="gray", markersize=10)

    st.pyplot(fig)


# Plot the soccer field with players
plot_players_on_field(selected_players, formation)
