import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

# Set up Streamlit page with custom theme
st.set_page_config(
    page_title="Fantasy Premier League",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
APP_TITLE = "Fantasy Premier League"
BUDGET = 1000  # Adjusted to match FPL's £100.0m as 1000 (since costs are in tenths of millions)

# Define formations with positions and required players
FORMATION_MAP = {
    '3-4-3': {'GKP': 1, 'DEF': 3, 'MID': 4, 'FWD': 3},
    '3-5-2': {'GKP': 1, 'DEF': 3, 'MID': 5, 'FWD': 2},
    '4-4-2': {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2},
    '4-3-3': {'GKP': 1, 'DEF': 4, 'MID': 3, 'FWD': 3},
}

# Field coordinates for player positions in different formations
FIELD_COORDS = {
    '3-4-3': {
        'GKP': [(40, 10)],
        'DEF': [(20, 30), (40, 30), (60, 30)],
        'MID': [(10, 60), (30, 60), (50, 60), (70, 60)],
        'FWD': [(20, 90), (40, 90), (60, 90)],
    },
    '3-5-2': {
        'GKP': [(40, 10)],
        'DEF': [(20, 30), (40, 30), (60, 30)],
        'MID': [(10, 60), (25, 60), (40, 60), (55, 60), (70, 60)],
        'FWD': [(30, 90), (50, 90)],
    },
    '4-4-2': {
        'GKP': [(40, 10)],
        'DEF': [(10, 30), (30, 30), (50, 30), (70, 30)],
        'MID': [(10, 60), (30, 60), (50, 60), (70, 60)],
        'FWD': [(30, 90), (50, 90)],
    },
    '4-3-3': {
        'GKP': [(40, 10)],
        'DEF': [(10, 30), (30, 30), (50, 30), (70, 30)],
        'MID': [(20, 50), (40, 50), (60, 50)],
        'FWD': [(10, 90), (40, 90), (70, 90)],
    },
}

# Position-specific metrics
POSITION_METRICS = {
    'GKP': ['saves', 'clean_sheets', 'goals_conceded'],
    'DEF': ['goals_scored', 'assists', 'clean_sheets', 'goals_conceded'],
    'MID': ['goals_scored', 'assists', 'clean_sheets'],
    'FWD': ['goals_scored', 'assists']
}

# Common metrics for all positions
COMMON_METRICS = [
    'id', 'web_name', 'position', 'now_cost', 'total_points', 'minutes',
    'yellow_cards', 'red_cards', 'bonus', 'bps', 'influence', 'creativity',
    'threat', 'selected_by_percent', 'form', 'points_per_game', 'in_dreamteam', 'dreamteam_count'
]

# App Title
st.title(APP_TITLE)

# Load and preprocess data
@st.cache_data
def load_player_data_from_api():
    """Fetches player data from the FPL API and returns a DataFrame with selected columns."""
    try:
        response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"There was an error: {e} while retrieving data")
        return pd.DataFrame()  # Return empty DataFrame on error

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
    df_final = df_merged[columns_to_use]

    return df_final

# Load player data
player_data = load_player_data_from_api()

# Function Definitions

def adjust_selected_players(position_counts):
    """Adjusts the selected players to match the new formation."""
    for position, count in position_counts.items():
        # Get current selected players for the position
        selected = st.session_state.selected_players.get(position, [])

        # If we need to reduce the number of players
        if len(selected) > count:
            # Keep only the required number of players
            selected = selected[:count]
            st.session_state.selected_players[position] = selected
        # If we need to add more players
        elif len(selected) < count:
            # Get available players not already selected
            selected_names = [p['web_name'] for p in selected]
            available_players = player_data[
                (player_data['position'] == position) &
                (~player_data['web_name'].isin(selected_names))
            ]
            # Select additional players based on total points
            needed = count - len(selected)
            top_players = available_players.sort_values(by='total_points', ascending=False).head(needed)
            selected.extend(top_players.to_dict('records'))
            st.session_state.selected_players[position] = selected
        # If the number is the same, do nothing

def select_players_for_position(position, count):
    """Handles player selection for a specific position."""
    # Widget key
    widget_key = f"select_{position}"

    # Filter available players
    available_players = player_data[player_data['position'] == position]
    options = available_players['web_name'].tolist()

    # Initialize default selections
    if widget_key not in st.session_state:
        # Get previously selected players for this position
        default_selections = [
            p['web_name'] for p in st.session_state.selected_players.get(position, [])
        ]
    else:
        # Use the selections from the session state
        default_selections = st.session_state[widget_key]

    # Ensure default selections do not exceed 'count' items
    if len(default_selections) > count:
        default_selections = default_selections[:count]
        st.session_state[widget_key] = default_selections  # Update session state before widget creation

    # Player selection multiselect
    st.sidebar.multiselect(
        f"Select {count} {position} player(s)",
        options=options,
        default=default_selections,
        key=widget_key,
        help=f"Select exactly {count} {position} player(s).",
        max_selections=count  # Requires Streamlit 1.22 or newer
    )

    # Retrieve selected names from session state
    selected_names = st.session_state[widget_key]

    # Update selected players in session_state after the widget
    selected_players = available_players[available_players['web_name'].isin(selected_names)].to_dict('records')
    st.session_state.selected_players[position] = selected_players

    return selected_players

@st.cache_data
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

@st.cache_data
def adjust_team_to_budget(team, budget):
    """Adjusts the team to fit within the budget by replacing expensive players."""
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

def draw_soccer_field(selected_team, formation):
    """Draws a soccer field with players positioned according to the formation."""
    field_color = "#6cba7c"  # Soft Grass Green
    line_color = "#ffffff"   # White lines

    # Field dimensions
    field_width = 80
    field_height = 120

    # Create Plotly figure
    fig = go.Figure()

    # Set layout
    fig.update_layout(
        xaxis=dict(
            range=[0, field_width],
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=True,
            domain=[0, 1],  # Fill the entire width
        ),
        yaxis=dict(
            range=[0, field_height],
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=True,
            domain=[0, 1],  # Fill the entire height
        ),
        width=700,    # Set the figure width
        height=1000,   # Set the figure height (adjust as needed)
        margin=dict(l=0, r=0, t=0, b=0),  # Remove all margins
        plot_bgcolor=field_color,
        paper_bgcolor=field_color,
        showlegend=False,
    )

    # Field boundary
    fig.add_shape(type="rect", x0=0, x1=field_width, y0=0, y1=field_height,
                  line=dict(color=line_color, width=2), layer='below')

    # Center line
    fig.add_shape(type="line", x0=0, y0=field_height / 2, x1=field_width, y1=field_height / 2,
                  line=dict(color=line_color, width=2), layer='below')

    # Penalty areas and other field markings
    # Bottom penalty box
    fig.add_shape(type="rect", x0=18, x1=62, y0=0, y1=18,
                  line=dict(color=line_color, width=2), layer='below')
    # Top penalty box
    fig.add_shape(type="rect", x0=18, x1=62, y0=102, y1=120,
                  line=dict(color=line_color, width=2), layer='below')

    # Goal areas
    # Bottom goal box
    fig.add_shape(type="rect", x0=30, x1=50, y0=0, y1=6,
                  line=dict(color=line_color, width=2), layer='below')
    # Top goal box
    fig.add_shape(type="rect", x0=30, x1=50, y0=114, y1=120,
                  line=dict(color=line_color, width=2), layer='below')

    # Center circle
    fig.add_shape(type="circle", x0=30, x1=50, y0=51, y1=69,
                  xref="x", yref="y",
                  line=dict(color=line_color, width=2), layer='below')
    # Center spot
    fig.add_shape(type="circle", x0=39.9, x1=40.1, y0=59.9, y1=60.1,
                  xref="x", yref="y",
                  fillcolor=line_color, line=dict(color=line_color, width=2), layer='below')

    # Penalty spots
    # Bottom penalty spot
    fig.add_shape(type="circle", x0=39.9, x1=40.1, y0=11.9, y1=12.1,
                  xref="x", yref="y",
                  fillcolor=line_color, line=dict(color=line_color, width=2), layer='below')
    # Top penalty spot
    fig.add_shape(type="circle", x0=39.9, x1=40.1, y0=107.9, y1=108.1,
                  xref="x", yref="y",
                  fillcolor=line_color, line=dict(color=line_color, width=2), layer='below')

    # Add players to the field
    coords = FIELD_COORDS[formation]
    for position, spots in coords.items():
        players = [p for p in selected_team if p['position'] == position]
        for i, (x, y) in enumerate(spots):
            if i < len(players):
                player = players[i]
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="markers+text",
                    marker=dict(size=25, color="#e90052", line=dict(width=2, color="#04f5ff")),
                    text=player['web_name'],
                    textposition="bottom center",
                    textfont=dict(size=11, color="#ffffff", family="Arial"),
                    hovertemplate=(
                        f"<b>{player['web_name']}</b><br>"
                        f"Position: {player['position']}<br>"
                        f"Team: {player['team_name']}<br>"
                        f"Cost: £{player['now_cost']/10:.1f}m<br>"
                        f"Points: {player['total_points']}<extra></extra>"
                    ),
                    showlegend=False
                ))
            else:
                # Placeholder markers
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="markers",
                    marker=dict(size=20, color="#808080", line=dict(width=2, color="#ffffff")),
                    hoverinfo="skip",
                    showlegend=False
                ))

    return fig

def plot_total_points_comparison(user_team, best_team):
    """Plots a bar chart comparing total points between two teams using consistent colors."""
    import plotly.express as px

    # Calculate total points for each team
    user_total_points = sum(player['total_points'] for player in user_team)
    best_total_points = sum(player['total_points'] for player in best_team)

    # Prepare data
    points_df = pd.DataFrame({
        'Team': ['Your Team', 'Best Team'],
        'Total Points': [user_total_points, best_total_points]
    })

    # Define the color mapping
    TEAM_COLOR_MAP = {
        'Your Team': '#e90052',
        'Best Team': '#04f5ff'
    }

    # Create the bar chart using Plotly Express
    fig = px.bar(
        points_df,
        x='Team',
        y='Total Points',
        text='Total Points',
        color='Team',
        title='Total Points Comparison',
        color_discrete_map=TEAM_COLOR_MAP
    )

    # Update the layout and traces
    fig.update_traces(
        texttemplate='%{text:.0f}',
        textposition='outside',
    )
    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        showlegend=False,  # Hide legend since team names are on the x-axis
        xaxis_title='',    # Remove x-axis title for a cleaner look
        yaxis_title='Total Points',
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


def display_team_table(team, team_name):
    """Displays the team information using Plotly's go.Table with enhanced styling."""

    st.subheader(team_name)
    team_df = pd.DataFrame(team)
    team_df['Cost (£m)'] = team_df['now_cost'] / 10  # Convert cost to millions

    data_frames = []

    # Define column rename mapping
    RENAME_COLUMNS = {
        'web_name': 'Player',
        'now_cost': 'Cost (£m)',
        'total_points': 'Total Points',
        'minutes': 'Minutes Played',
        'goals_scored': 'Goals Scored',
        'assists': 'Assists',
        'clean_sheets': 'Clean Sheets',
        'goals_conceded': 'Goals Conceded',
        'saves': 'Saves',
        'yellow_cards': 'Yellow Cards',
        'red_cards': 'Red Cards',
        'bonus': 'Bonus',
        'bps': 'BPS',
        'influence': 'Influence',
        'creativity': 'Creativity',
        'threat': 'Threat',
        'selected_by_percent': 'Selected By (%)',
        'form': 'Form',
        'points_per_game': 'Points Per Game',
        'team_name': 'Team',
        'in_dreamteam': 'In Dream Team',
        'dreamteam_count': 'Dreamteam Count',
        # 'photo_url': 'Photo',
    }

    for position in ['GKP', 'DEF', 'MID', 'FWD']:
        position_players = team_df[team_df['position'] == position]
        if not position_players.empty:
            # Select relevant metrics, including 'team_name'
            metrics = COMMON_METRICS + POSITION_METRICS[position] + ['team_name']
            position_df = position_players[metrics]

            # Rename columns
            position_df.rename(columns=RENAME_COLUMNS, inplace=True)

            # Add position column for clarity
            position_df['Position'] = position

            # Map position-specific metrics to their renamed versions
            position_metrics_renamed = [RENAME_COLUMNS.get(metric, metric) for metric in POSITION_METRICS[position]]

            # Reorder columns and include 'Points Per Game'
            columns_order = ['Player', 'Position', 'Team', 'Cost (£m)', 'Total Points',
                             'Points Per Game'] + position_metrics_renamed
            position_df = position_df[columns_order]

            data_frames.append(position_df)

    if data_frames:
        display_df = pd.concat(data_frames, ignore_index=True)
        display_df.fillna('', inplace=True)  # Fill missing values with empty strings

        # Convert data types if necessary
        numeric_columns = ['Cost (£m)', 'Total Points', 'Points Per Game'] + position_metrics_renamed
        for col in numeric_columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

        # Define color scales for conditional formatting
        def get_color(val, metric):
            """Assigns a color based on the value and metric."""
            if metric in ['Total Points', 'Points Per Game', 'Assists', 'Goals Scored']:
                if val >= 15:
                    return '#d4edda'  # Light green
                elif val >= 10:
                    return '#fff3cd'  # Light yellow
                else:
                    return '#f8d7da'  # Light red
            elif metric in ['Cost (£m)']:
                if val <= 5:
                    return '#d1ecf1'  # Light blue
                elif val <= 7:
                    return '#bee5eb'  # Medium blue
                else:
                    return '#abdde5'  # Darker blue
            else:
                return '#f9f9f9' if i % 2 == 0 else '#ffffff'  # Alternating row colors

        # Apply conditional coloring to cells
        cell_colors = []
        for i in range(len(display_df)):
            row_colors = []
            for col in display_df.columns:
                if col in ['Total Points', 'Points Per Game', 'Assists', 'Goals Scored', 'Cost (£m)']:
                    color = get_color(display_df.at[i, col], col)
                else:
                    color = '#f9f9f9' if i % 2 == 0 else '#ffffff'  # Alternating row colors
                row_colors.append(color)
            cell_colors.append(row_colors)

        # Create the Plotly table with enhanced styling
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_df.columns),
                fill_color='#343a40',  # Dark header background
                font=dict(color='white', size=12, family='Arial'),
                align='center',
                height=40,
                line=dict(color='#FFFFFF', width=2)  # White borders for headers
            ),
            cells=dict(
                values=[display_df[col] for col in display_df.columns],
                fill_color=cell_colors,
                font=dict(color='black', size=10, family='Arial'),
                align='center',
                height=30,
                line=dict(color='#FFFFFF', width=1)  # White borders for cells
            )
        )])

        fig.update_layout(
            width=1200,  # Increased width for better visibility
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
        )

        # Display the Plotly table in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No players to display.")


def plot_team_radar_chart(user_team, best_team):
    """Plots a radar chart comparing average metrics between two teams."""

    # Metrics to compare
    metrics = ['Goals Scored', 'Assists', 'Clean Sheets', 'Points Per Game', 'Selected By (%)']

    # Prepare data
    def get_team_averages(team):
        df = pd.DataFrame(team)
        df['Points Per Game'] = pd.to_numeric(df['points_per_game'], errors='coerce')
        df['Selected By (%)'] = pd.to_numeric(df['selected_by_percent'], errors='coerce')
        averages = {
            'Goals Scored': df['goals_scored'].mean(),
            'Assists': df['assists'].mean(),
            'Clean Sheets': df['clean_sheets'].mean(),
            'Points Per Game': df['Points Per Game'].mean(),
            'Selected By (%)': df['Selected By (%)'].mean(),
        }
        return averages

    user_averages = get_team_averages(user_team)
    best_averages = get_team_averages(best_team)

    categories = list(user_averages.keys())

    user_values = list(user_averages.values())
    best_values = list(best_averages.values())

    # Close the loop for radar chart
    categories += categories[:1]
    user_values += user_values[:1]
    best_values += best_values[:1]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Your Team',
        line_color='#e90052',
        opacity=0.7
    ))
    fig.add_trace(go.Scatterpolar(
        r=best_values,
        theta=categories,
        fill='toself',
        name='Best Team',
        line_color='#04f5ff',
        opacity=0.7
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(user_values), max(best_values)) * 1.1]
            )
        ),
        showlegend=True,
        title="Average Team Metrics Comparison",
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_points_contribution_by_position(user_team, best_team):
    """Plots a bar chart comparing points contribution by position between two teams."""

    # Function to calculate points per position
    def calculate_points_per_position(team):
        df = pd.DataFrame(team)
        points_per_position = df.groupby('position')['total_points'].sum().reset_index()
        return points_per_position

    # Calculate points per position for both teams
    user_ppp = calculate_points_per_position(user_team)
    best_ppp = calculate_points_per_position(best_team)

    # Rename columns for clarity
    user_ppp.rename(columns={'total_points': 'Your Team'}, inplace=True)
    best_ppp.rename(columns={'total_points': 'Best Team'}, inplace=True)

    # Merge the two DataFrames on 'position'
    merged_ppp = pd.merge(user_ppp, best_ppp, on='position', how='outer').fillna(0)

    # Melt the DataFrame for Plotly Express
    melted_ppp = merged_ppp.melt(id_vars='position', value_vars=['Your Team', 'Best Team'],
                                 var_name='Team', value_name='Total Points')

    # Sort positions for consistent ordering
    position_order = ['GKP', 'DEF', 'MID', 'FWD']
    melted_ppp['position'] = pd.Categorical(melted_ppp['position'], categories=position_order, ordered=True)
    melted_ppp.sort_values('position', inplace=True)

    # Define the color mapping
    TEAM_COLOR_MAP = {
        'Your Team': '#e90052',
        'Best Team': '#04f5ff'
    }

    # Create the bar chart
    fig = px.bar(
        melted_ppp,
        x='position',
        y='Total Points',
        color='Team',
        barmode='group',
        title='Points Contribution by Position',
        labels={'position': 'Position', 'Total Points': 'Total Points'},
        color_discrete_map=TEAM_COLOR_MAP
    )

    fig.update_layout(
        xaxis_title='Position',
        yaxis_title='Total Points',
        legend_title='Team',
        template='plotly'
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_cost_vs_points(user_team, best_team):
    """Plots a scatter plot of player cost vs. total points."""

    combined_team = user_team + best_team
    df = pd.DataFrame(combined_team)
    #df.rename(columns={'web_name':'Name'},inplace=True)
    df['Cost (£m)'] = df['now_cost'] / 10
    df['Team'] = ['Your Team'] * len(user_team) + ['Best Team'] * len(best_team)
    df['Total Points'] = pd.to_numeric(df['total_points'], errors='coerce')

    fig = px.scatter(
        df,
        x='Cost (£m)',
        y='Total Points',
        color='Team',
        hover_data=['web_name', 'position'],
        title='Player Cost vs. Total Points',
        color_discrete_map={'Your Team': '#e90052', 'Best Team': '#04f5ff'},
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

# Initialize session state
if 'selected_players' not in st.session_state:
    st.session_state.selected_players = {}
if 'best_team' not in st.session_state:
    st.session_state.best_team = None
if 'formation' not in st.session_state:
    st.session_state.formation = None

# Sidebar - Team Selection
st.sidebar.title("Team Selection")
formation = st.sidebar.selectbox("Select Formation", list(FORMATION_MAP.keys()), index=0)
position_counts = FORMATION_MAP[formation]

# Check if formation has changed
formation_changed = formation != st.session_state.formation
st.session_state.formation = formation

# Adjust selected players and widget state if formation changed
if formation_changed:
    adjust_selected_players(position_counts)
    st.session_state.best_team = None  # Reset best team

    # Adjust widget state for each position
    for position, count in position_counts.items():
        widget_key = f"select_{position}"

        # If the widget key exists in session state
        if widget_key in st.session_state:
            selected_names = st.session_state[widget_key]
            # Truncate the list if it has more items than allowed
            if len(selected_names) > count:
                st.session_state[widget_key] = selected_names[:count]

# Collect players by position
selected_players = []
total_cost = 0

for position, count in position_counts.items():
    position_players = select_players_for_position(position, count)
    position_cost = sum(int(player['now_cost']) for player in position_players)
    total_cost += position_cost
    selected_players.extend(position_players)

# Calculate remaining budget
remaining_budget = BUDGET - total_cost

# Display budget information
st.sidebar.write(f"**Total Cost:** £{total_cost/10:.1f}m / £{BUDGET/10:.1f}m")
st.sidebar.write(f"**Remaining Budget:** £{remaining_budget/10:.1f}m")
if total_cost > BUDGET:
    st.sidebar.error("Budget exceeded!")

# Compute the best team if necessary
if st.session_state.best_team is None or formation_changed:
    # Step 1: Get top players by position
    best_team = get_top_players_by_position(player_data, formation)
    # Step 2: Adjust team to fit within the budget
    best_team = adjust_team_to_budget(best_team, BUDGET)
    st.session_state.best_team = best_team
else:
    best_team = st.session_state.best_team



col1, col2 = st.columns([3, 1])
with col1:
    team_to_display = st.radio("Select Team to Display on Field", ['Your Team', 'Best Team'])

    if team_to_display == 'Your Team':
        team_to_show = selected_players
    else:
        team_to_show = best_team

    field_fig = draw_soccer_field(team_to_show, formation)
    st.plotly_chart(field_fig, use_container_width=True)

with col2:
    # Display total cost comparison
    user_total_cost = sum(player['now_cost'] for player in selected_players)
    best_total_cost = sum(player['now_cost'] for player in best_team)

    st.write(f"**Your Team Cost:** £{user_total_cost / 10:.1f}m / £{BUDGET / 10:.1f}m")
    st.write(f"**Best Team Cost:** £{best_total_cost / 10:.1f}m / £{BUDGET / 10:.1f}m")

    if user_total_cost > BUDGET:
        st.error("Your team's budget is exceeded!")

    if best_total_cost > BUDGET:
        st.warning("The best team exceeds the budget constraints.")



# Team Comparison and Details
st.subheader("Team Comparison")
plot_total_points_comparison(selected_players, best_team)

plot_team_radar_chart(selected_players, best_team)

plot_points_contribution_by_position(selected_players, best_team)

plot_cost_vs_points(selected_players, best_team)


# Team Details in Tabs
st.subheader("Team Details")

team_tabs = st.tabs(["Your Team", "Best Team"])

with team_tabs[0]:
    display_team_table(selected_players, "Your Team")

with team_tabs[1]:
    display_team_table(best_team, "Best Team")

# Highlight common players
user_player_names = set(player['web_name'] for player in selected_players)
best_player_names = set(player['web_name'] for player in best_team)
common_players = user_player_names & best_player_names

if common_players:
    st.write(f"**Common Players:** {', '.join(common_players)}")
else:
    st.write("**No common players between your team and the best team.**")

