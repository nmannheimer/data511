# team_selection.py

import streamlit as st
from constants import FORMATION_MAP, POSITION_FULL_NAMES, COLOR_PALETTE, SECTION_ICONS

def adjust_selected_players(position_counts, player_data):
    """
    Adjusts the selected players in session state to match the new formation.

    Parameters:
    - position_counts (dict): Dictionary mapping positions to required counts.
    - player_data (pd.DataFrame): DataFrame containing player statistics.
    """
    for position, count in position_counts.items():
        # Get current selected players for the position
        selected = st.session_state.selected_players.get(position, [])

        # If we need to reduce the number of players
        if len(selected) > count:
            # Keep only the required number of players
            selected = selected[:count]
            st.session_state.selected_players[position] = selected
        # If we need to add more players, do NOT auto-populate or display warnings
        elif len(selected) < count:
            # Do not auto-populate; user must select manually
            st.session_state.selected_players[position] = selected
        # If the number is the same, do nothing

def select_players_for_position(position, count, player_data):
    """
    Handles player selection for a specific position with enforced selection limits.

    Parameters:
    - position (str): The position to select players for (e.g., 'DEF').
    - count (int): The number of players to select for the position.
    - player_data (pd.DataFrame): DataFrame containing player statistics.

    Returns:
    - selected_players (list of dict): List of selected player dictionaries.
    """
    # Widget key
    widget_key = f"select_{position}"

    # Filter available players
    available_players = player_data[player_data['position'] == position]
    options = available_players['web_name'].tolist()

    # Initialize default selections as empty
    default_selections = []

    # Retrieve the full position name for the label
    full_position_name = POSITION_FULL_NAMES.get(position, position)

    # Player selection multiselect with max_selections
    selected_names = st.sidebar.multiselect(
        f"### {SECTION_ICONS['Pick Players']} Pick Your {full_position_name}",
        options=options,
        default=default_selections,
        key=widget_key,
        help=f"Select exactly {count} {full_position_name}.",
        max_selections=count  # Enforce selection limit
    )

    # Update selected players in session_state after the widget
    selected_players = available_players[available_players['web_name'].isin(selected_names)].to_dict('records')
    st.session_state.selected_players[position] = selected_players

    return selected_players