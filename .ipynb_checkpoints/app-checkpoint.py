import streamlit as st
import pandas as pd
import requests

st.title("Fantasy Premier League")

st.write("Welcome to Fantasy Premier League!")

def get_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status






