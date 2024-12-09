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
        'DEF': [(20, 25), (40, 25), (60, 25)],
        'MID': [(10, 35), (30, 35), (50, 35), (70, 35)],
        'FWD': [(20, 45), (40, 45), (60, 45)],
    },
    '3-5-2': {
        'GKP': [(40, 10)],
        'DEF': [(20, 25), (40, 25), (60, 25)],
        'MID': [(10, 35), (25, 35), (40, 35), (55, 35), (70, 35)],
        'FWD': [(30, 45), (50, 45)],
    },
    '4-4-2': {
        'GKP': [(40, 10)],
        'DEF': [(10, 25), (30, 25), (50, 25), (70, 25)],
        'MID': [(10, 35), (30, 35), (50, 35), (70, 35)],
        'FWD': [(30, 45), (50, 45)],
    },
    '4-3-3': {
        'GKP': [(40, 10)],
        'DEF': [(10, 25), (30, 25), (50, 25), (70, 25)],
        'MID': [(20, 35), (40, 35), (60, 35)],
        'FWD': [(20, 45), (40, 45), (60, 45)],
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
    'App Title': '#d90050',      # Pinkish Red
    'Sidebar Pick': '#4ff1fe',    # Light Blue
    'Sidebar Budget': '#ebff00',  # Bright Yellow
    'Performance Analysis': '#4eff83',  # Vibrant Green
    'Predicted Points' : '#A020F0',
    'Gray': '#aaaaaa',             # Gray for neutral elements
    'Black': '#000000'
}

# Combined Color Palette
COMBINED_COLOR_PALETTE = [
    # Core Application Colors (Brand-Related)
    '#f5005f',  # Pinkish Red (App Title)
    '#4ff1fe',  # Light Blue (Sidebar Pick)
    '#ebff00',  # Bright Yellow (Sidebar Budget)
    '#4eff83',  # Vibrant Green (Performance Analysis)
    '#A020F0',  # Purple (Predicted Points)
    '#aaaaaa',  # Gray (Neutral Elements)
    '#04f5ff',  # Light Blue (Best Team)#
    '#e90052',  # Red (Your Team)
    '#d90050',  # Pinkish Red (App Title)

    # Position-Based Colors
    '#636EFA',  # Blue (GKP)
    '#EF553B',  # Red (DEF)
    '#00CC96',  # Green (MID)
    '#AB63FA'   # Purple (FWD)
]

# Existing SECTION_ICONS
SECTION_ICONS = {
    'App Title': '⚽️',
    'Pick Players': '📋',
    'Budget Overview': '💰',
    'Performance Analysis': '📊',
    'Metrics Radar': '📈',
    'Cost Distribution': '💸',
    'Shared Players': '🔍',
    'Target' : '🎯',
    # Add the following keys:
    'Your Team': '👤',       # Represents the user's team
    'Best Team': '🏆'        # Represents the best possible team
}