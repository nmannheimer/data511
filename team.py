# main.py

import streamlit as st
from data_loader import load_player_data_from_api
from team_selection import adjust_selected_players, select_players_for_position
from team_computation import get_top_players_by_position, adjust_team_to_budget
from visualizations import (
    draw_soccer_field,
    plot_total_points_comparison,
    plot_team_radar_chart,
    plot_cost_breakdown_by_position
)
from constants import FORMATION_MAP, BUDGET, COLOR_PALETTE, SECTION_ICONS, POSITION_FULL_NAMES


def main():
    # Set up Streamlit page with custom theme
    st.set_page_config(
        page_title="Ultimate FPL Manager",
        page_icon="⚽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # App Title with Color and Icon
    st.markdown(
        f"<h1 style='text-align: center; color: {COLOR_PALETTE['App Title']};'>{SECTION_ICONS['App Title']} Ultimate FPL Manager</h1>",
        unsafe_allow_html=True
    )

    # Load player data
    player_data = load_player_data_from_api()

    if player_data.empty:
        st.stop()  # Stop execution if data could not be loaded

    # Initialize session state
    if 'selected_players' not in st.session_state:
        st.session_state.selected_players = {position: [] for position in ['GKP', 'DEF', 'MID', 'FWD']}
    if 'best_team' not in st.session_state:
        st.session_state.best_team = None
    if 'formation' not in st.session_state:
        st.session_state.formation = None

    # Sidebar - Team Selection
    st.sidebar.markdown(
        f"<h2 style='color: {COLOR_PALETTE['Sidebar Pick']};'>{SECTION_ICONS['Pick Players']} Build Your Dream Team</h2>",
        unsafe_allow_html=True
    )
    formation = st.sidebar.selectbox("Choose Your Formation", list(FORMATION_MAP.keys()), index=0)
    position_counts = FORMATION_MAP[formation]

    # Check if formation has changed
    formation_changed = formation != st.session_state.formation
    st.session_state.formation = formation

    # Adjust selected players and widget state if formation changed
    if formation_changed:
        adjust_selected_players(position_counts, player_data)
        st.session_state.best_team = None  # Reset best team

        # Adjust widget state for each position by resetting selections
        for position, count in position_counts.items():
            widget_key = f"select_{position}"
            st.session_state.selected_players[position] = []

    # Collect players by position
    selected_players = []
    total_cost = 0

    for position, count in position_counts.items():
        position_players = select_players_for_position(position, count, player_data)
        position_cost = sum(int(player['now_cost']) for player in position_players)
        total_cost += position_cost
        selected_players.extend(position_players)

    # Calculate remaining budget
    remaining_budget = BUDGET - total_cost

    # Final Validation: Check if all positions have required players
    all_positions_complete = all(len(st.session_state.selected_players.get(position, [])) >= count
                                 for position, count in position_counts.items())

    if not all_positions_complete:
        st.sidebar.error("Please ensure you have selected all required players for each position.")

    # Check if budget is exceeded
    if total_cost > BUDGET:
        st.sidebar.error("Budget exceeded!")

    # Compute the best team if necessary
    if st.session_state.best_team is None or formation_changed:
        # Step 1: Get top players by position
        best_team = get_top_players_by_position(player_data, formation)
        # Step 2: Adjust team to fit within the budget
        best_team = adjust_team_to_budget(best_team, BUDGET, player_data)
        st.session_state.best_team = best_team
    else:
        best_team = st.session_state.best_team

    # Display Teams and Visualizations
    col1, col2 = st.columns([2, 1])  # Allocate ~75% width to the field and ~25% to additional info
    with col1:
        team_to_display = st.radio("Select Team to Visualize", ['Your Team', 'Best Team'])

        if team_to_display == 'Your Team':
            team_to_show = selected_players
        else:
            team_to_show = best_team

        if not team_to_show:
            st.write("**No players selected. Please select your team to view the field.**")
            field_fig = draw_soccer_field([], formation)
        else:
            field_fig = draw_soccer_field(team_to_show, formation)
        st.plotly_chart(field_fig, use_container_width=True)

    with col2:
        # Display total cost comparison
        user_total_cost = sum(player['now_cost'] for player in selected_players)
        best_total_cost = sum(player['now_cost'] for player in best_team)

        st.markdown(
            f"<h3 style='color: {COLOR_PALETTE['Sidebar Budget']};'>{SECTION_ICONS['Budget Overview']} Budget Overview</h3>",
            unsafe_allow_html=True
        )
        st.write(f"**Your Team Cost:** £{user_total_cost / 10:.1f}m / £{BUDGET / 10:.1f}m")
        st.write(f"**Best Team Cost:** £{best_total_cost / 10:.1f}m / £{BUDGET / 10:.1f}m")

        if user_total_cost > BUDGET:
            st.error("Your team's budget is exceeded!")

        if best_total_cost > BUDGET:
            st.warning("The best team exceeds the budget constraints.")

        # ***Added Section: Display Player Photos Aligned with Formation***
        st.markdown(
            f"<h4 style='color: {COLOR_PALETTE['Performance Analysis']};'>{SECTION_ICONS['Performance Analysis']} {team_to_display} Players</h4>",
            unsafe_allow_html=True
        )

        if team_to_show:
            # Group players by position in the order of the formation
            positions_order = ['FWD', 'MID', 'DEF', 'GKP']
            for pos in positions_order:
                pos_players = [player for player in team_to_show if player['position'] == pos]
                if pos_players:
                    # Create columns for each player in the position
                    cols = st.columns(len(pos_players))
                    for idx, player in enumerate(pos_players):
                        with cols[idx]:
                            # Display player image; use a default image if 'photo_url' is missing
                            photo_url = player.get('photo_url', 'https://via.placeholder.com/100')  # Placeholder image
                            st.image(photo_url, width=80)
                            st.caption(player['web_name'])
        else:
            st.write("**Please select your team or best team to view player photos.**")

    # Team Comparison and Details
    st.markdown(
        f"<h3 align='center' style='color: {COLOR_PALETTE['Performance Analysis']};'>{SECTION_ICONS['Performance Analysis']} Team Performance and Cost Analysis</h3>",
        unsafe_allow_html=True
    )

    st.markdown (
        f"<h4 style='color: {COLOR_PALETTE['Performance Analysis']};'>{SECTION_ICONS['Performance Analysis']} Total Points Comparison</h4>",
        unsafe_allow_html=True)
    plot_total_points_comparison(selected_players, best_team)

    st.markdown(
        f"<h4 style='color: {COLOR_PALETTE['Performance Analysis']};'>{SECTION_ICONS['Performance Analysis']} Average Team Performance Metrics Comparison</h4>",
        unsafe_allow_html=True
    )
    plot_team_radar_chart(selected_players, best_team)

    st.markdown(
        f"<h4 style='color: {COLOR_PALETTE['Performance Analysis']};'>{SECTION_ICONS['Cost Distribution']} Position-wise Cost Distribution</h4>",
        unsafe_allow_html=True)
    plot_cost_breakdown_by_position(selected_players, best_team)

    # Highlight common players
    user_player_names = set(player['web_name'] for player in selected_players)
    best_player_names = set(player['web_name'] for player in best_team)
    common_players = user_player_names & best_player_names

    if common_players:
        st.write(f"**{SECTION_ICONS['Shared Players']} Shared Players Spotlight:** {', '.join(common_players)}")
    else:
        st.write("**No common players between your team and the best team.**")


if __name__ == "__main__":
    main()