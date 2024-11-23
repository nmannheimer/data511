import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from streamlit_elements import elements, mui, html, nivo

import streamlit as st
from streamlit_elements import elements, nivo

# Sample data for the pie chart
data = [
    {"id": "JavaScript", "label": "JavaScript", "value": 55},
    {"id": "Python", "label": "Python", "value": 75},
    {"id": "Java", "label": "Java", "value": 30},
    {"id": "C#", "label": "C#", "value": 20},
    {"id": "PHP", "label": "PHP", "value": 15}
]

# Streamlit sidebar input for customizing chart data
st.sidebar.header("Pie Chart Settings")
chart_title = st.sidebar.text_input("Chart Title", "Programming Language Popularity")
inner_radius = st.sidebar.slider("Inner Radius", 0.0, 1.0, 0.5)

# Display the pie chart using Streamlit Elements
st.title(chart_title)

with elements("nivo_pie_chart"):
    with mui.Box(sx={"height": 500}):
        nivo.Pie(
            data=data,
            margin={"top": 40, "right": 80, "bottom": 80, "left": 80},
            innerRadius=0.5,
            padAngle=0.7,
            cornerRadius=3,
            colors={ "scheme": "nivo" },
            borderWidth=1,
            borderColor={"from": "color", "modifiers": [["darker", 0.2]]},
            radialLabelsSkipAngle=10,
            radialLabelsTextColor="#333333",
            radialLabelsLinkColor={"from": "color"},
            sliceLabelsSkipAngle=10,
            sliceLabelsTextColor="#333333"
        )
