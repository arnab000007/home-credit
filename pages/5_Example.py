import streamlit as st
import pandas as pd
import numpy as np
from Utility.PlotingUtility import PlotingUtility
import seaborn as sns

placeholder = st.empty()
isclick = placeholder.button('delete this button')
if isclick:
    placeholder.empty()