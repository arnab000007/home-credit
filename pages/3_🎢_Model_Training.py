import streamlit as st
import pandas as pd
import numpy as np
from prettytable import PrettyTable

st.set_page_config(
    page_title="Model Trainings",
    page_icon=":roller_coaster:",
)

st.title('Home Credit Model Trainings')


st.markdown('''
We have started simple model like KNN Classifier and then we have trained more complex model like Random Forest and GBDT based ensumble model.
After all the training we have found **LGBClassifier(GBDT based)** have given the highest AUC score on validation and test data.
''')

st.markdown('''
    To Train the LGBClassifier, we have used following hyperparameters. We have used a custom randomized search alogorithm to train the LGBClassifier.
    
     - **num_leaves** - [20,25,30,31,32,35,40,45,50,55,60,65,70,75]
     - **max_depth** - [3,5,10,12,13,15,18,20,25,30,40,50]
     - **learning_rate** - [0.0001,0.0003,0.001,0.003, 0.01,0.03,0.1]
     - **n_estimators** - [40,60,100,150,200,500,750,900,1000,1250,1500,1750,2000,2500,3000]
     - **reg_alpha** - np.logspace(-4, 2, 18)
     - **reg_lambda** - np.logspace(-4, 2, 18)
     - **colsample_bytree** - [ 0.2,0.4,0.5,0.6,0.8]
     - **subsample** - [0.5,0.6,0.7,0.8,0.9]

''')

st.markdown('''
    These are the valudation AUC score for different values of hyperparammeters. 
''')
tbl = PrettyTable(['num_leaves', "max_depth", "learning_rate", "n_estimators", 'reg_alpha','reg_lambda','colsample_bytree','subsample','Rank','Val AUC'])
tbl.add_row([30,30,0.03,900,0.0131,0.1501,0.2,0.5,1,0.78949])
tbl.add_row([30,15,0.1,750,1.7191,0.3384,0.2,0.9,2,0.78876])
tbl.add_row([70,13,0.1,1000,0.0005,0.0131,0.6,0.6,3,0.78607])
tbl.add_row([50,40,0.003,1750,0.0666,0.058,0.2,0.6,4,0.78115])
tbl.add_row([40,15,0.01,1750,8.7333,44.3669,0.5,0.5,5,0.77615])

st.write(tbl)
st.markdown('')
st.markdown('''

    Here is the Freature Importance graph for most useful feature. 

''')
st.image('images/feature_importances.png', caption='Feature Importance Plot')

st.markdown('''
    Here we have seen that some freature which we have created in feature engineering step have very good freture importance values. 
    The ROC plot of Train and Validation data is given below.
''')
st.image('images/roc.png', caption='ROC Plot for Train and Validation data')