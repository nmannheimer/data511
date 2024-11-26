# visualizations.py

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit import title

from constants import FIELD_COORDS_HALF, POSITION_COLORS, COMMON_METRICS, POSITION_METRICS, POSITION_FULL_NAMES
import pandas as pd
import streamlit as st


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
            range=[0, field_height],
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=True,
            domain=[0, 1],  # Fill the entire height
        ),
        width=700,  # Adjusted figure width for good aspect ratio
        height=600,  # Adjusted figure height for half-field
        margin=dict(l=0, r=0, t=0, b=0),  # Remove all margins
        plot_bgcolor=field_color,
        paper_bgcolor=field_color,
        showlegend=False,
    )

    # Draw field boundaries and markings (only half-field)
    # Field boundary
    fig.add_shape(type="rect", x0=0, x1=field_width, y0=0, y1=field_height,
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
    fig.add_shape(type="circle", x0=30, x1=50, y0=31, y1=49,
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
                        f"Points: {player['total_points']}<extra></extra>"
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
        xaxis_title='',  # Remove x-axis title for a cleaner look
        yaxis_title='Total Points',
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
        df['Points Per Game'] = pd.to_numeric(df['points_per_game'], errors='coerce').fillna(0)
        df['Selected By (%)'] = pd.to_numeric(df['selected_by_percent'], errors='coerce').fillna(0)
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
        template='plotly_dark'
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
                df = df.append({'position': pos, 'now_cost': 0}, ignore_index=True)
        return df

    user_cpp = ensure_all_positions(user_cpp)
    best_cpp = ensure_all_positions(best_cpp)

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=('Your Team', 'Best Team')
    )

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
        legend_title="Position",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50, b=150)  # Adjust margins to accommodate legends
    )

    # Display the pie charts
    st.plotly_chart(fig, use_container_width=True)


