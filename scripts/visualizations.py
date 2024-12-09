# visualizations.py
from tempfile import template

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit import title
import pandas as pd
from constants import COLOR_PALETTE, POSITION_COLORS, FORMATION_MAP, FIELD_COORDS_HALF, POSITION_COLORS, COMMON_METRICS, POSITION_METRICS, POSITION_FULL_NAMES
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.express as px
import pandas as pd
import os
from pathlib import Path
import sys

# player_pred_file = Path(os.getcwd()).parent.as_posix() + '/data/predicted_df.csv'
#players_pred_df = pd.read_csv('./predicted_df.csv')

players_pred_df = pd.read_csv("../data/predicted_df.csv")

def get_player_pred(name, team):
    try:
        name_clean = name.strip().split(".")[-1]
        x = players_pred_df[players_pred_df.web_name.str.contains(name_clean)][players_pred_df.team == team]
        return int(x.sort_values(['gw'], ascending = False).pred_points_rounded.iloc[0])
    except:
        return 0


def draw_soccer_field(selected_team, formation):
    """Draws a half soccer field with players positioned according to the formation."""
    field_color = "#6cba7c"  # Soft Grass Green
    line_color = "#ffffff"  # White lines

    # Field dimensions for half-field
    field_width = 80
    field_height = 80  # Half of the original height

    # Create Plotly figure
    fig = go.Figure()

    # Set layout with good aspect ratio
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
            range=[0, field_height*.75],
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=True,
            domain=[0, 1],  # Fill the entire height
        ),
        width=650,  # Adjusted figure width for good aspect ratio
        height=750,  # Adjusted figure height for half-field
        margin=dict(l=0, r=0, t=0, b=0),  # Remove all margins
        plot_bgcolor=field_color,
        paper_bgcolor=field_color,
        showlegend=False,
    )

    # Draw field boundaries and markings (only half-field)
    # Field boundary
    fig.add_shape(type="rect", x0=0, x1=field_width, y0=field_height*.75, y1=field_height*.75,
                  line=dict(color=line_color, width=2), layer='below')

    # Center line (midfield line)
    fig.add_shape(type="line", x0=0, y0=field_height / 2, x1=field_width, y1=field_height / 2,
                  line=dict(color=line_color, width=2), layer='below')

    # Penalty areas and other field markings (half-field)
    # Bottom penalty box
    fig.add_shape(type="rect", x0=18, x1=62, y0=0, y1=18,
                  line=dict(color=line_color, width=2), layer='below')

    # Goal areas
    # Bottom goal box
    fig.add_shape(type="rect", x0=30, x1=50, y0=0, y1=6,
                  line=dict(color=line_color, width=2), layer='below')

    # Center circle
    fig.add_shape(type="circle", x0=35, x1=45, y0=37, y1=43,
                  xref="x", yref="y",
                  line=dict(color=line_color, width=2), layer='below')
    # Center spot
    fig.add_shape(type="circle", x0=39.9, x1=40.1, y0=39.9, y1=40.1,
                  xref="x", yref="y",
                  fillcolor=line_color, line=dict(color=line_color, width=2), layer='below')

    # Penalty spots
    # Bottom penalty spot
    fig.add_shape(type="circle", x0=39.9, x1=40.1, y0=11.9, y1=12.1,
                  xref="x", yref="y",
                  fillcolor=line_color, line=dict(color=line_color, width=2), layer='below')

    # Add players to the field
    coords = FIELD_COORDS_HALF[formation]  # Use half-field coordinates
    for position, spots in coords.items():
        players = [p for p in selected_team if p['position'] == position]
        for i, (x, y) in enumerate(spots):
            if i < len(players):
                player = players[i]
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(
                        size=25,
                        color=POSITION_COLORS.get(position, "#e90052"),  # Original color fill
                        line=dict(width=2, color="#04f5ff")
                    ),
                    text=player['web_name'],
                    textposition="bottom center",
                    textfont=dict(size=11, color="#ffffff", family="Arial"),
                    hovertemplate=(
                        f"<b>{player['web_name']}</b><br>"
                        f"Position: {player['position']}<br>"
                        f"Team: {player['team_name']}<br>"
                        f"Cost: £{player['now_cost'] / 10:.1f}m<br>"
                        f"Points: {player['total_points']}<br>"
                        f"Expected Points next GW: {get_player_pred(player['web_name'], player['team_name'])}<extra></extra>"
                    ),
                    showlegend=False
                ))
            else:
                # Placeholder for empty spots
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color="#808080",
                        line=dict(width=2, color="#ffffff")
                    ),
                    hoverinfo="skip",
                    showlegend=False
                ))

    return fig



def plot_total_points_comparison(user_team, best_team):
    """Plots a bar chart comparing total points between two teams using consistent colors."""
    # Handle empty teams by initializing total points to zero
    if not user_team:
        user_total_points = 0
    else:
        user_total_points = sum(player['total_points'] for player in user_team)

    if not best_team:
        best_total_points = 0
    else:
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
        title='Your Team vs. Best Team Total Points Comparison',
        color_discrete_map=TEAM_COLOR_MAP
    )

    # Update the layout and traces
    fig.update_traces(
        texttemplate='%{text:.0f}',
        textposition='outside',
        textfont=dict(color='black')
    )
    fig.update_layout(
        yaxis=dict(titlefont=dict(color="black"), title='Total Points', tickfont=dict(color="black")),
        xaxis=dict(titlefont=dict(color="black"), tickfont=dict(color="black")),
        titlefont=dict(color="black", size=14, family="Arial"),
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        showlegend=False,  # Hide legend since team names are on the x-axis
        paper_bgcolor='#E0FEFF',
        plot_bgcolor='#E0FEFF',
        title={
            "text": 'Average Metrics Comparison',
            "x": 0.5, "xanchor": "center",
            "y": 0.9, "yanchor": "top"}
    )


    # Display the chart
    st.plotly_chart(fig, use_container_width=True)



def plot_team_radar_chart(user_team, best_team):
    """Plots a radar chart comparing average metrics between two teams."""

    # Metrics to compare
    metrics = ['Goals Scored', 'Assists', 'Clean Sheets', 'Points Per Game', 'Selected By (%)']

    # Prepare data
    def get_team_averages(team):
        if not team:
            # Initialize averages with zeros if team is empty
            return {metric: 0 for metric in metrics}
        df = pd.DataFrame(team)
        df['Points Per Game'] = pd.to_numeric(df['points_per_game'], errors='coerce').fillna(0)/10
        df['Selected By (%)'] = pd.to_numeric(df['selected_by_percent'], errors='coerce').fillna(0)/100

        averages = {
            'Goals Scored': df['goals_scored'].mean()/10,
            'Assists': df['assists'].mean()/7,
            'Clean Sheets': df['clean_sheets'].mean()/8,
            'Points Per Game': df['Points Per Game'].mean(),
            'Selected By (%)': df['Selected By (%)'].mean()
        }

        # Extract the values and compute min and max
        vals = np.array(list(averages.values()))
        min_val = vals.min()
        max_val = vals.max()

        # Perform min-max normalization
        normalized_averages = {
            k: (v - min_val) / (max_val - min_val) for k, v in averages.items()
        }

        return normalized_averages

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
        opacity=0.7,
        textfont = dict(color='red')

    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(user_values), max(best_values)) * 1.1]
            ),
            angularaxis=dict(
                tickfont=dict(color="black")
            )

        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="right",
            x=1,
            font=dict(color="black")
        ),
        titlefont=dict(color="black", size=14, family="Arial"),
        template='plotly_white',
        paper_bgcolor = '#E0FEFF',
        plot_bgcolor = '#E0FEFF',
        title={
            "text": 'Total Points Comparison',
            "x": 0.5, "xanchor": "center",
            "y": 0.9, "yanchor": "top"},
        font=dict(color="black")


    )

    st.plotly_chart(fig, use_container_width=True)



def plot_cost_breakdown_by_position(user_team, best_team):
    """Plots pie charts showing cost breakdown by position for user team and best team."""

    # Function to calculate cost per position
    def calculate_cost_per_position(team):
        if not team:
            # Initialize with zero costs for all positions
            positions = ['GKP', 'DEF', 'MID', 'FWD']
            return pd.DataFrame({
                'position': positions,
                'now_cost': [0, 0, 0, 0]
            })
        df = pd.DataFrame(team)
        cost_per_position = df.groupby('position')['now_cost'].sum().reset_index()
        return cost_per_position

    # Calculate cost per position for both teams
    user_cpp = calculate_cost_per_position(user_team)
    best_cpp = calculate_cost_per_position(best_team)

    # Ensure all positions are present in the DataFrame
    def ensure_all_positions(df):
        all_positions = ['GKP', 'DEF', 'MID', 'FWD']
        for pos in all_positions:
            if pos not in df['position'].values:

                new_row = pd.DataFrame([{'position': pos, 'now_cost': 0}])
                df = pd.concat([df, new_row], ignore_index=True)
        return df

    user_cpp = ensure_all_positions(user_cpp)
    best_cpp = ensure_all_positions(best_cpp)

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=('Your Team', 'Best Team'),

    )
    for annotation in fig['layout']['annotations']:
        annotation['font']['color'] = 'black'

    # Add pie chart for user team
    fig.add_trace(
        go.Pie(
            labels=user_cpp['position'].map(POSITION_FULL_NAMES),  # Map abbreviations to full names
            values=user_cpp['now_cost'] / 10,  # Convert to millions
            marker=dict(colors=[POSITION_COLORS.get(pos, '#808080') for pos in user_cpp['position']]),
            name='Your Team',
            hoverinfo='label+percent+value',
            textinfo='label+percent',
            textfont=dict(color='#000000')
        ),
        row=1, col=1
    )

    # Add pie chart for best team
    fig.add_trace(
        go.Pie(
            labels=best_cpp['position'].map(POSITION_FULL_NAMES),
            values=best_cpp['now_cost'] / 10,  # Convert to millions
            marker=dict(colors=[POSITION_COLORS.get(pos, '#808080') for pos in best_cpp['position']]),
            name='Best Team',
            hoverinfo='label+percent+value',
            textinfo='label+percent',
            textfont=dict(color='#000000')
        ),
        row=1, col=2
    )


    # Update layout for aesthetics
    fig.update_layout(
        showlegend=True,  # Enable legends
        legend_title="Field Position",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="right",
            x=0.8,
            title_font = dict(color='black', size=14, family='Arial'),  # Change legend title color and size
            font = dict(color='black')
    ),
        paper_bgcolor='#E0FEFF',
        plot_bgcolor='#E0FEFF',
        template = 'plotly_white',
        margin=dict(t=100, b=150),  # Adjust margins to accommodate legends
        title = {"text":"Your Team vs Best Team Cost Distribution",
                 "x": 0.5, "xanchor": "center", "y": 0.9, "yanchor": "top"},
        titlefont = dict(color="black")
    )

    # Display the pie charts
    st.plotly_chart(fig, use_container_width=True)


def total_points_vs_cost_yearly(df: pd.DataFrame, min_minutes: int = 500):
    """Plots a scatter plot of Points Scored vs Cost that can dynamically be adjusted based on position and cost."""

    # Step 1: Filter DataFrame by minimum minutes played
    filtered_df = df[df["minutes"] > min_minutes]
    filtered_df["now_cost_m"] = filtered_df["now_cost"] / 10  # Convert cost to millions

    # Positions
    positions = filtered_df['position'].unique().tolist()
    fig = go.Figure()

    # Step 2: Create scatter plot with POSITION_COLORS
    for pos in positions:
        position_data = filtered_df[filtered_df['position'] == pos]
        fig.add_trace(
            go.Scatter(
                x=position_data['now_cost_m'],
                y=position_data['total_points'],
                mode='markers',
                name=pos,
                marker=dict(
                    size=9,
                    opacity=0.8,
                    line=dict(width=1, color='white'),
                    color=POSITION_COLORS.get(pos, COLOR_PALETTE['Gray'])  # Fallback to Gray if not found
                ),
                customdata=position_data[['web_name', 'position']],
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    'Position: %{customdata[1]}<br>'
                    'Cost: £%{x:.1f}M<br>'
                    'Points: %{y}<extra></extra>'
                )
            )
        )

    # Step 3: Create dropdown for position filtering
    dropdown_buttons = [
        dict(
            label="All Positions",
            method="update",
            args=[
                {"visible": [True] * len(fig.data)},  # Show all traces
                {"title": "Player Points vs. Cost (All Positions)"}
            ]
        )
    ]

    for i, pos in enumerate(positions):
        dropdown_buttons.append(
            dict(
                label=pos,
                method="update",
                args=[
                    {"visible": [trace.name == pos for trace in fig.data]},
                    {"title": f"Player Points vs. Cost ({pos})"}
                ]
            )
        )

    # Step 4: Add slider for max cost filtering
    max_cost = int(filtered_df['now_cost_m'].max())
    min_cost = int(filtered_df['now_cost_m'].min())

    steps = []
    for cost in range(min_cost, max_cost + 1):
        step = dict(
            method="restyle",
            args=[
                {
                    "x": [trace.x[trace.x <= cost] if trace.x is not None else [] for trace in fig.data],
                    "y": [trace.y[trace.x <= cost] if trace.y is not None else [] for trace in fig.data]
                }
            ],
            label=f"{cost}M"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Max Cost: £", "font": {"color": "black", "size": 12}},
        pad={"t": 50, "b": 20},
        steps=steps,
        transition={"duration": 0},
        lenmode="fraction",
        len=1.0,
        bgcolor='#d90050',
        bordercolor="#ccc",
        borderwidth=1
    )]

    # Step 5: Update layout with dropdown and sliders
    fig.update_layout(
        sliders=sliders,
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.8,
                y=1,
                bgcolor="#9EFDFF",
                bordercolor='#d90050',
                borderwidth=1,
                font=dict(color="black")
            )
        ],
        title={
            "text": "Player Points vs. Cost in FPL",
            "x": 0.5, "xanchor": "center",
            "y": 0.9, "yanchor": "top",
            "font": {"color": "black", "size": 14}
        },
        xaxis=dict(
            title="Cost (in £ millions)",
            tickformat='.1f',
            gridcolor='gray',
            zerolinecolor='gray',
            linecolor='black',
            titlefont=dict(color="black"),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title="Total Points Scored",
            gridcolor='gray',
            zerolinecolor='gray',
            linecolor='black',
            titlefont=dict(color="black"),
            tickfont=dict(color='black')
        ),
        #height=700,
        #width=1000,
        paper_bgcolor='#E0FEFF',
        plot_bgcolor='#E0FEFF',

        font=dict(
            family="Arial, sans-serif",
            color='black',
            size=14
        ),
        template='plotly_white',
    )

    st.plotly_chart(fig, use_container_width=True)
    
def plot_gw_performance_by_player(player_name: str, df: pd.DataFrame):
    """Plot the performance of a player every gameweek."""
    
    player_df = df[df["name"] == player_name] 
    fig = px.line(
        data_frame = player_df,
        x = 'GW',
        y = 'total_points',
        title = f"⚽ {player_name}'s Points Over Each Gameweek 🏟️",
        labels = {'GW' : 'Gameweeks', 'total_points': 'Points Earned'},
        hover_data = {'goals_scored': True, 'assists': True, 'minutes' : True},
        markers = True
    )

    fig.update_traces(
        line = dict(color = '#AB63FA', width = 3, dash = 'solid'),
        marker = dict(size = 10, color = '#AB63FA', symbol = 'circle'),
        hovertemplate = (
            '<b>Gameweek %{x}</b><br>'
            'Points: %{y}<br>'
            'Goals ⚽: %{customdata[0]}<br>'
            'Assists 🅰️: %{customdata[1]}<br>'
            'Minutes Played ⏱️: %{customdata[2]}<br>'
        ),
        customdata = player_df[['goals_scored', 'assists', 'minutes']]
    )

    fig.update_layout(
        plot_bgcolor='#E0FEFF',  # Football-themed black background
        paper_bgcolor='#E0FEFF',
        font=dict(color='black'),
        # font = dict(color = 'white', size = 14),
        title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        title_font=dict(size=14, color='black', family='Arial Black'),
        xaxis=dict(
            gridcolor='gray',
            linecolor='gray',
            tickfont=dict(color='black'),
            titlefont=dict(color='black'),
        ),
        yaxis=dict(
            gridcolor='gray',
            linecolor='gray',
            tickfont=dict(color='black'),
            rangemode='tozero',
            titlefont=dict(color='black'),
        ),
        height = 500,
        width = 600,
        template='plotly_white',
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_transfers_in_out_by_player(player_name: str, df: pd.DataFrame):
    """Plots the transfers in vs transfers out of a player every gameweek."""

    player_df = df[df["name_cleaned"] == player_name]
    fig = go.Figure()

    # Add Transfers In line
    fig.add_trace(
        go.Scatter(
            x=player_df['GW'],
            y=player_df['transfers_in'],
            mode='lines+markers',
            name='Transfers In',
            line=dict(color='#04f5ff', width=3),
            marker=dict(size=8)
        )
    )

    # Add Transfers Out line
    fig.add_trace(
        go.Scatter(
            x=player_df['GW'],
            y=player_df['transfers_out'],
            mode='lines+markers',
            name='Transfers Out',
            line=dict(color='#e90052', width=3),
            marker=dict(size=8)
        )
    )

    # Update layout
    fig.update_layout(
        xaxis=dict(title='Gameweek', tickmode='linear', gridcolor='gray', titlefont=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title='Transfers', gridcolor='gray', titlefont=dict(color='black'), tickfont=dict(color='black'), range=[0, player_df['transfers_in'].max() + 2]),
        height=600,
        width=600,
        plot_bgcolor='#E0FEFF',  # Football-themed black background
        paper_bgcolor='#E0FEFF',
        font=dict(color='black'),
        title_font=dict(size=14, color='black', family='Arial'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color='black')
        ),
        template='plotly_white',
        title={ "text": f"Transfers In and Out Per Gameweek: \n{player_name}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },

    )

    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
    
def radar_chart_player_comparison(df: pd.DataFrame, player1: str, player2: str, metrics: list):
    """
    Create a radar chart to compare two players across selected metrics.
    
    Parameters:
        df (pd.DataFrame): The dataset containing player stats.
        player1 (str): The name of the first player.
        player2 (str): The name of the second player.
        metrics (list): List of metric columns to compare. This should ideally vary between different positions.
    """

    # Metric to label mapping
    METRIC_LABELS = {
        'total_points': 'Total Points',
        'minutes': 'Minutes Played',
        'goals_scored': 'Goals Scored',
        'assists': 'Assists',
        'clean_sheets': 'Clean Sheets',
        'goals_conceded': 'Goals Conceded',
        'selected_by_percent': 'Ownership (%)'
    }
     # Step 1: Ensure numeric columns for the metrics
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')

    # Step 2: Normalize the metrics between 0 and 1
    normalized_df = df.copy()
    for metric in metrics:
        min_val = normalized_df[metric].min()
        max_val = normalized_df[metric].max()
        normalized_df[metric] = (normalized_df[metric] - min_val) / (max_val - min_val)

    # Step 3: Filter data for the two players
    players_df = normalized_df[normalized_df['full_name'].isin([player1, player2])]

    # Step 4: Filter only the relevant metrics and player name
    players_df = players_df[['full_name'] + metrics]

    # Step 5: Reshape the data for radar plotting
    melted_df = players_df.melt(id_vars='full_name', var_name='metric', value_name='value')

    # Map the 'metric' column values to user-friendly labels
    melted_df['metric'] = melted_df['metric'].apply(lambda x: METRIC_LABELS.get(x, x))

    # Step 6: Create radar chart
    fig = px.line_polar(
        melted_df,
        r='value',
        theta='metric',
        color='full_name',
        line_close=True,
        title=f"{player1} vs {player2}",
        template="plotly_white"
    )

    # Customize layout
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),  # Normalized range
            angularaxis=dict(showline=True, tickfont=dict(size=12))
        ),
        title_font=dict(size=14, family='Arial', color='black'),
        legend=dict(title = "Players", title_font=dict(size=12, family='Arial', color='black'), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color='black')),
        paper_bgcolor='#E0FEFF',
        plot_bgcolor='#E0FEFF',
        font=dict(color='black'),
        showlegend=True,
        margin = dict(b=50),
        template='plotly_white',
        title={
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }

    )

    st.plotly_chart(fig, use_container_width=True)
    
def top_n_roi_by_position(df: pd.DataFrame, pos:str, top_n:int = 5):
    """
    Calculates the ROI of a player per 90 minutes, filtered for minutes played greater than 400 mins. 
    Returns a bar chart of top_n players per position.
    """
    
    df["points/90"] = round((df["total_points"]/df["minutes"])*90, 3).fillna(0)
    df["ROI"] = round(df["points/90"]/df["now_cost_m"],3).fillna(0)
    
    filtered_df = df[(df["minutes"] > 400) & (df["position"] == pos)].sort_values(by = ["ROI"], ascending=False)[:5]
    
    fig = px.bar(
        filtered_df,
        x = 'web_name',
        y = 'ROI',
        text = 'ROI',
        title=f"Top 5 ROI Players by {pos} position",
        labels={'ROI': 'ROI (Points per Million)', 'name': 'Player Name', 'total_points': 'Total Points'},
        height=1000,
        width=700
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        showlegend=False,
        uniformtext_minsize=8,
        uniformtext_mode='hide',
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_fpl_performance_funnel(df, players, player='full_name', total_points_column='total_points', xp_column='xP'):
    # Set the background to black
    plt.style.use('seaborn')
    # Filter the dataframe for the players in the list
    df_filtered = df[df[player].isin(players)]
    # If the filtered dataframe is empty, inform the user
    if df_filtered.empty:
        print(f"Error: No players found matching the names in the list {players}")
        return
    # Calculate the residuals (actual points - expected points)
    df_filtered['Residual'] = df_filtered[total_points_column] - df_filtered[xp_column]
    # Calculate mean and standard deviation of the residuals for each player
    residual_stats = df_filtered.groupby(player)['Residual'].agg(['mean', 'std']).reset_index()
    # Rename columns to match the original dataframe's player column name
    residual_stats.rename(columns={'mean': 'mean_residual', 
                                   'std': 'std_residual'}, inplace=True)
    # Merge the residual statistics back with the original filtered dataframe
    df_filtered = pd.merge(df_filtered, residual_stats[[player, 'mean_residual', 'std_residual']], 
                           on=player, how='left')
    # Define custom colors for the first two players (green, red) and others (random colors)
    colors = ['#04f5ff', '#e90052', 'white']
    
    fig, ax = plt.subplots(figsize=(5,5), frameon=False)
    custom_palette = sns.color_palette(['#04f5ff', '#e90052'])
    # Scatter plot of actual points (total_points) vs expected points (xP)
    sns.scatterplot(data=df_filtered, 
                    x=xp_column, 
                    y=total_points_column, 
                    hue=player,      # Color by player name
                    style=player,    # Different marker for each player
                    palette=custom_palette,   # Choose a palette for colors
                    markers='o',      # Use circle markers (default)
                    size='Residual',  # Size by residual
                    sizes=(20, 200),  # Adjust size range
                    alpha=0.8, 
                    ax=ax)        # Set transparency for better visibility
    # Set distinct colors for players, with the first two being green and red
    color_map = sns.color_palette("Set2", len(players))  # Generate a color palette for the players
    player_colors = {players[i]: colors[i] for i in range(len(players))}  # Assign colors
    # Plot the funnel plot bounds for each player with different colors
    for player_name, group in df_filtered.groupby(player):
        player_mean = group['mean_residual'].iloc[0]
        
        # Get the player's color from the color_map
        player_color = player_colors.get(player_name, 'gray')
        # Plot bounds for each player with their specific color
        ax.axhline(player_mean, color=player_color, linestyle='--', label=f'{player_name} Mean Residual')
    # Labels and title
    ax.set_title("FPL Performance Funnel Plot:\nActual vs Expected Points", fontsize=12, color='white')
    ax.set_xlabel(f"Expected Points ({xp_column})", fontsize=10, color='white')
    ax.set_ylabel(f"Total Points ({total_points_column})", fontsize=10, color='white')
    ax.legend(title="Player Performance", loc="upper left", bbox_to_anchor=(0.01, -0.25), frameon=False, labelcolor='white')
    st.pyplot(fig, use_container_width=False)

    

def ownership_vs_points_bubble_chart_with_dropdown(df: pd.DataFrame, min_ownership_pct: float):
    """
    Create a bubble chart with a dropdown to filter by player position.
    Parameters:
        df (pd.DataFrame): The dataset containing player stats.
        min_ownership_pct (float): The maximum ownership percentage for filtering players.
    """
    # Step 1: Ensure required columns are numeric
    df["now_cost_m"] = df["now_cost"]/10
    df["points/90"] = round((df["total_points"]/df["minutes"])*90, 3).fillna(0)
    df["ROI"] = round(df["points/90"]/df["now_cost_m"],3).fillna(0)
    df['selected_by_percent'] = pd.to_numeric(df['selected_by_percent'], errors='coerce')
    df['total_points'] = pd.to_numeric(df['total_points'], errors='coerce')
    df['now_cost_m'] = pd.to_numeric(df['now_cost_m'], errors='coerce')
    df['ROI'] = pd.to_numeric(df['ROI'], errors='coerce')
    # Step 2: Get unique positions
    positions = df['position'].unique()
    # Step 3: Filter data for each position
    filtered_data = {}
    for pos in positions:
        filtered_data[pos] = df[
            (df['position'] == pos) &
            (df['selected_by_percent'] < min_ownership_pct) &
            (df['selected_by_percent'] > 2)
        ]
    # Step 4: Create the initial figure for the first position
    initial_position = positions[0]
    fig = px.scatter(
        filtered_data[initial_position],
        x='selected_by_percent',
        y='ROI',
        size='now_cost_m',  # Bubble size based on cost
        color='position',
        hover_name='full_name',
        title=f"ROI for {initial_position} players for Ownership less than {min_ownership_pct}%",
        labels={
            'selected_by_percent': 'Ownership Percentage (%)',
            'ROI': 'ROI',
            'now_cost_m': 'Cost (in £M)'
        },
        template='plotly_white',
        #height=600,
        #width=900
    )
    # Customize bubble size
    fig.update_traces(
        marker=dict(
            sizeref=2. * df['now_cost_m'].max() / (10 ** 2),  # Adjust this to scale bubbles down
            sizemin=5,  # Minimum bubble size
            opacity=0.8,
            sizemode='diameter'
        )
    )
    # Step 5: Add dropdown for position filtering
    dropdown_buttons = []
    for pos in positions:
        dropdown_buttons.append(
            dict(
                label=pos,
                method="update",
                args=[
                    {
                        "x": [filtered_data[pos]['selected_by_percent']],
                        "y": [filtered_data[pos]['ROI']],
                        "marker.size": [filtered_data[pos]['now_cost_m']]
                    },
                    {"title": f"ROI for {pos} players for Ownership less than {min_ownership_pct}%"
                     }
                ]
            )
        )
    # Step 6: Add dropdown menu to layout
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.9,
                y=1.15,
                xanchor="left",
                yanchor="top",
                bgcolor='#9EFDFF',
                bordercolor='#d90050',
                font = dict(color='black')
            )

        ],
        xaxis=dict(title="Ownership Percentage (%)", titlefont = dict(color='black'), tickfont = dict(color='black')),
        yaxis=dict(title="ROI", titlefont = dict(color='black'), tickfont = dict(color='black')),
        legend=dict(title="Position"),
        coloraxis_colorbar=dict(title="Position"),
        paper_bgcolor='#E0FEFF',
        plot_bgcolor='#E0FEFF',
        titlefont = dict(color='black'),
        title = {
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }

    )
    # Show the chart
    st.plotly_chart(fig, use_container_width=False)
    
def plot_player_vs_avg_actual_points(df, full_name):
    # Filter the data for the specific player
    player_data = df[df['name'] == full_name]
    # Find the player's position
    player_position = player_data['position'].iloc[0]
    # Calculate the average actual points for the player's position
    avg_position_data = df[df['position'] == player_position]
    avg_actual_points = avg_position_data.groupby('GW')['total_points'].mean().reset_index()
    # Set the dark theme for the plot
    plt.style.use('dark_background')
    # Create a figure to plot the bar chart and lines
    fig, ax = plt.subplots(figsize=(12, 8))
    # Loop through the gameweeks and conditionally color the bars based on 'was_home'
    for i, row in player_data.iterrows():
        bar_color = '#EF553B' if row['was_home'] == 1 else '#636EFA'  # Blue for home, Red/Pink for away
        ax.bar(row['GW'], row['total_points'], width=0.4, color=bar_color, label=f'{full_name} - Actual Points' if i == 0 else "")
    # Line plot for the average actual points for the position
    ax.plot(avg_actual_points['GW'], avg_actual_points['total_points'], label=f'Average {player_position} - Actual Points', color='#BA55D3', linestyle='--', linewidth=2)
    # Add labels and title
    ax.set_xlabel('Gameweek', color='white')
    ax.set_ylabel('Actual Points', color='white')
    ax.set_title(f'{full_name} Actual Points vs. Average {player_position} Performance Over Time', color='white')
    # Create a custom legend for home/away
    home_away_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E90FF', markersize=10, label='Home'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6347', markersize=10, label='Away')
    ]
    # Add the legend to the plot
    ax.legend(handles=home_away_legend + ax.get_legend_handles_labels()[0])
    # Customize ticks and grid
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    # Show the plot
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)