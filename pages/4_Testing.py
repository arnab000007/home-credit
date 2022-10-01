import streamlit as st
import pandas as pd
import numpy as np
import pickle

from lightgbm import LGBMClassifier

st.title('Home Credit Testing the Models')

@st.cache(allow_output_mutation=True)
def load_model():
    with open('Models/NewFeatureMinMax_LGB_Cal.pkl' , 'rb') as f:
        clf = pickle.load(f)
        return clf


@st.cache(allow_output_mutation=True)
def load_data(num_rows = 100):
    xt = pd.read_csv('Data/application_test.csv', nrows=num_rows)
    xtf = pd.read_csv('Data/X_Test_Final_OHE_New_All.csv')
    xtf = xtf.loc[list(xt.index)]
    return xt,xtf

st.sidebar.header("Testing the Model")

X_Test, X_Test_Final = load_data(100)

option = st.sidebar.selectbox(
    'Please Select a User',
    X_Test['SK_ID_CURR'].values)

xt_curr = X_Test[X_Test['SK_ID_CURR']== option]

st.markdown('**The data of the selected user given below.**')
st.markdown('**Selected User** - {}'.format(option))
st.markdown('#### This is the original data')
st.dataframe(xt_curr,use_container_width=True)


xtf_curr = X_Test_Final.loc[list(xt_curr.index)]
st.markdown('#### This is transform data of the selected user')
st.dataframe(xtf_curr,use_container_width=True)

sig_clf = load_model()

st.markdown('**The Probability of Default for the user is {}**'.format(sig_clf.predict_proba(xtf_curr)[:,1][0]))



