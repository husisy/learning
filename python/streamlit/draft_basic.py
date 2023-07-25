import time
import pandas as pd
import numpy as np

import streamlit as st

np_rng = np.random.default_rng()

## streamlit run draft_basic.py
st.title('streamlit demo')


# regenerated every time web page is refreshed
st.write('`st.dataframe(np)` can be sorted and edited')
np0 = np_rng.integers(0, 10, size=(4,3))
st.dataframe(np0)

# static table
st.write('`st.table(np)` cannot be sorted and edited')
st.table(np0)


st.write('`st.dataframe(pd.io.formats.style.Styler)`')
tmp0 = np_rng.uniform(0, 1, size=(6,4))
tmp1 = [f'col {x}' for x in range(tmp0.shape[1])]
pd0 = pd.DataFrame(tmp0, columns=tmp1)
tmp2 = pd0.style.highlight_max(axis=0) #pandas.io.formats.style.Styler
st.dataframe(tmp2)


st.write('`st.line_chart(pd.DataFrame)`')
tmp0 = np.linspace(0, 2*np.pi, 50)
pd0 = pd.DataFrame({
    'sin': np.sin(tmp0),
    'cos': np.cos(tmp0),
})
st.line_chart(pd0)
# st.write(pd0)


st.write('`st.map(pd.DataFrame)`')
np0 = np_rng.normal(scale=0.02, size=(1000, 2)) + np.array([37.76, -122.4])
pd0 = pd.DataFrame(np0, columns=['latitude', 'longitude']) #predfined column names
st.map(pd0)


st.write('`st.slider(int)`')
z0 = st.slider('x', min_value=0, max_value=100)
st.write(z0, 'squared is', z0*z0)
st.write('`st.slider(float)`')
z1 = st.slider('x', min_value=0.0, max_value=100.0, value=25.0) #value=(25.0, 75.0)
st.write(z1, 'squared is', z1*z1)


st.write('`st.write(session_state)`')
st.text_input('Your name', key='name')
if st.session_state.name:
    st.write('hello ', st.session_state.name)


st.write('`st.latex(str)`')
st.latex('E=mc^2')
st.write('`str.write(str)` inline latex: $E=mc^2$')


st.write('`st.checkbox(str)`')
if st.checkbox('show dataframe'):
    st.dataframe(np_rng.integers(0, 10, size=(3,3)))


st.write('`st.selectbox(str, list)`')
pd0 = pd.DataFrame({
    'column 1': [1, 2, 3, 4],
    'column 2': [10, 20, 30, 40]
})
option = st.selectbox('select a item:', pd0['column 1'])
st.write('you selected:', option)


st.sidebar.write('`st.sidebar.xxx`')
option = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

st.write('`st.columns()`')
left_column, right_column = st.columns(2)
left_column.button('press me')
with right_column:
    chosen = st.radio('sorting hat', ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")


st.write('`st.progress(int)`')
botton = st.button('start progress')
if botton:
    latest_iteration = st.empty()
    bar = st.progress(0) #0-100
    for i in range(10):
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress((i + 1)*10)
        time.sleep(0.4)


st.write('`st.session_state`')
COUNTER_NAME = '_counter00'
if COUNTER_NAME not in st.session_state:
    st.session_state[COUNTER_NAME] = 0
def hf_counter00_onclick():
    st.session_state[COUNTER_NAME] += 1
button = st.button('counter increment', on_click=hf_counter00_onclick)
st.write('Counter=', st.session_state[COUNTER_NAME])
