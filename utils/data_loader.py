# data_loader.py

import pandas as pd
import requests
import streamlit as st
from utils.constants import COMMON_METRICS
import unicodedata

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
    df_final.loc[:, 'full_name_pre'] = df_final['first_name'] + ' ' + df_final['second_name']
    # Remove accents from player_names
    
    # Function to remove accents
    def remove_accents(input_str):
        return ''.join(
            char for char in unicodedata.normalize('NFD', input_str)
            if unicodedata.category(char) != 'Mn'
        )

    # Apply the function to the 'web_name' column
    df_final['full_name'] = df_final['full_name_pre'].apply(remove_accents)

    return df_final

@st.cache_data()
def load_gameweek_data_from_github(year: str):
    """Fetches gameweek by gameweek player data from the Github Dataset and returns a DataFrame with selected columns."""
    
    try:
        url_gw = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{year}/gws/merged_gw.csv"
        df = pd.read_csv(url_gw)
    except Exception as e:
        st.error(f"There was an error: {e} while retrieving data")
        return pd.DataFrame()
    
    df["position"] = df["position"].apply(lambda x: 'GKP' if x == 'GK' else x)
    
    # Function to remove accents
    def remove_accents(input_str):
        return ''.join(
            char for char in unicodedata.normalize('NFD', input_str)
            if unicodedata.category(char) != 'Mn'
        )

    # Apply the function to the 'web_name' column
    df['name_cleaned'] = df['name'].apply(remove_accents)
    
    return df