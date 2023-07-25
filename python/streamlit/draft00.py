import time
import numpy as np
import streamlit as st

# streamlit run --server.port=23333 --server.headless=true draft01.py

def hf_set_seed():
    try:
        seed = int(st.session_state['input_seed'])
    except:
        seed = int(np.random.default_rng().integers(int(1e18)))
    st.session_state['seed'] = str(seed)
    st.session_state['np_rng'] = np.random.default_rng(seed)
tmp0 = 'Set random seed (input empty to randomize)'
tmp1 = st.session_state.get('seed', None)
if tmp1 is None:
    tmp1 = str(np.random.default_rng().integers(int(1e18)))
    st.session_state['np_rng'] = np.random.default_rng(int(tmp1))
st.text_input(tmp0, tmp1, key='input_seed', on_change=hf_set_seed)


@st.cache_data
def load_time_consuming_data(num_data):
    np_rng = np.random.default_rng(seed=233) #seed in st.cache_data is ignored
    time.sleep(2)
    tmp0 = np_rng.normal(0, 5, size=(num_data))
    ret = np.round(tmp0).astype(np.int64) % 24
    return ret

st.subheader('Demo st.cache_data')
st.text_input('number of data:', value=233, key='num_data')
try:
    num_data = int(st.session_state.num_data)
except:
    st.text('not a valid number, using 233')
    num_data = 233
tmp0 = st.text('loading data...')
data = load_time_consuming_data(num_data)
tmp0.text('loading data...done!')
hist_values = np.histogram(data, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)
