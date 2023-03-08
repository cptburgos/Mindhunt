from constants import (BOOTSTRAP, BANNER, FOOTER)
import streamlit as st
import pandas as pd
import numpy as np

st.markdown(BOOTSTRAP, unsafe_allow_html=True)
st.markdown(BANNER, unsafe_allow_html=True)

chart_data = pd.DataFrame(
    np.random.rand(20, 3),
    columns = ['a', 'b', 'c'])

st.line_chart(chart_data)

df_map = pd.DataFrame(
    np.random.rand(100, 2) / [40,40] + [13.7, -89.21],
    columns = ['lat', 'lon'])

st.map(df_map)


st.markdown(FOOTER, unsafe_allow_html=True)