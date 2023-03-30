import streamlit as st
import pandas as pd
import numpy as np


st.write('\n# My first app\n## with streamlit')

df = pd.DataFrame({'data': np.random.standard_normal(23)})
st.line_chart(df)
# streamlit run app.py
