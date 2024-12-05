import streamlit as st
from constants import FORMATION_MAP, POSITION_FULL_NAMES, COLOR_PALETTE, SECTION_ICONS


def adjust_selected_players(position_counts, player_data):
    for position, count in position_counts.items():
        selected = st.session_state.selected_players.get(position, [])

        # If more players are selected than allowed, truncate the list
        if len(selected) > count:
            selected = selected[:count]
            st.session_state.selected_players[position] = selected

            # Also sync the widget state to prevent the multiselect from restoring too many selections.
            widget_key = f"select_{position}"
            st.session_state[widget_key] = [p['web_name'] for p in selected]

        # If fewer players than needed, leave as is (user will pick more)


def select_players_for_position(position, count, player_data):
    widget_key = f"select_{position}"
    available_players = player_data[player_data['position'] == position]
    options = available_players['web_name'].tolist()
    full_position_name = POSITION_FULL_NAMES.get(position, position)

    # Do not provide a default; rely entirely on session_state.
    selected_names = st.sidebar.multiselect(
        f"### {SECTION_ICONS['Pick Players']} Pick Your {full_position_name}",
        options=options,
        key=widget_key,
        help=f"Select exactly {count} {full_position_name}.",
        max_selections=count
    )

    selected_players = available_players[available_players['web_name'].isin(selected_names)].to_dict('records')
    st.session_state.selected_players[position] = selected_players
    return selected_players