# constants.py

# App Constants
APP_TITLE = "Fantasy Premier League"
BUDGET = 1000  # Represents £100.0m (since costs are in tenths of millions)

# Define formations with positions and required players
FORMATION_MAP = {
    '3-4-3': {'GKP': 1, 'DEF': 3, 'MID': 4, 'FWD': 3},
    '3-5-2': {'GKP': 1, 'DEF': 3, 'MID': 5, 'FWD': 2},
    '4-4-2': {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2},
    '4-3-3': {'GKP': 1, 'DEF': 4, 'MID': 3, 'FWD': 3},
}

# Field coordinates for player positions in different formations (Half Field)
FIELD_COORDS_HALF = {
    '3-4-3': {
        'GKP': [(40, 10)],
        'DEF': [(20, 30), (40, 30), (60, 30)],
        'MID': [(10, 50), (30, 50), (50, 50), (70, 50)],
        'FWD': [(20, 70), (40, 70), (60, 70)],
    },
    '3-5-2': {
        'GKP': [(40, 10)],
        'DEF': [(20, 30), (40, 30), (60, 30)],
        'MID': [(10, 50), (25, 50), (40, 50), (55, 50), (70, 50)],
        'FWD': [(30, 70), (50, 70)],
    },
    '4-4-2': {
        'GKP': [(40, 10)],
        'DEF': [(10, 30), (30, 30), (50, 30), (70, 30)],
        'MID': [(10, 50), (30, 50), (50, 50), (70, 50)],
        'FWD': [(30, 70), (50, 70)],
    },
    '4-3-3': {
        'GKP': [(40, 10)],
        'DEF': [(10, 30), (30, 30), (50, 30), (70, 30)],
        'MID': [(20, 50), (40, 50), (60, 50)],
        'FWD': [(10, 70), (40, 70), (70, 70)],
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

# Position colors for visualizations
POSITION_COLORS = {
    'GKP': '#636EFA',  # Blue
    'DEF': '#EF553B',  # Red
    'MID': '#00CC96',  # Green
    'FWD': '#AB63FA'   # Purple
}

POSITION_FULL_NAMES = {
    'GKP': 'Goalkeeper',
    'DEF': 'Defenders',
    'MID': 'Midfielders',
    'FWD': 'Forwards'
}



# Color Palette
COLOR_PALETTE = {
    'App Title': '#f5005f',      # Pinkish Red
    'Sidebar Pick': '#4ff1fe',    # Light Blue
    'Sidebar Budget': '#ebff00',  # Bright Yellow
    'Performance Analysis': '#4eff83',  # Vibrant Green
    'Gray': '#aaaaaa'             # Gray for neutral elements
}

# Icon Mapping
# constants.py

# Existing SECTION_ICONS
SECTION_ICONS = {
    'App Title': '⚽️',
    'Pick Players': '📋',
    'Budget Overview': '💰',
    'Performance Analysis': '📊',
    'Metrics Radar': '📈',
    'Cost Distribution': '💸',
    'Shared Players': '🔍',
    # Add the following keys:
    'Your Team': '👤',       # Represents the user's team
    'Best Team': '🏆'        # Represents the best possible team
}