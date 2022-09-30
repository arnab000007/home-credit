import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Feature Engineering",
    page_icon=":wrench:"
)

st.title('Home Credit Feature Engineering')

st.write('We have tried some feature engineering and all details are given below.')

st.markdown('''
    #### Model based imputation for EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3
    
    We have seen that these 3 features have very good feature importance while we trained a Random Forest or GBDT based ensumble model. Using each feature,
    we have created 2 features one this the original one and null values are imputed with mean values and another one is created based on the LGBMRegressor.

    To train the LGBMRegressor model, we have used all other features. 

    - We have seen that for Column EXT_SOURCE_1 - Number of null values for train data: 138803, Validation Data: 34575, Test Data: 20532.
    
    - For Column EXT_SOURCE_2 - Number of null values for train data: 544, Validation Data: 116, Test Data: 8
    
    - For Column EXT_SOURCE_3 - Number of null values for train data: 48728, Validation Data: 12237, Test Data: 8668

    For the feature EXT_SOURCE_1 and EXT_SOURCE_3, there are some NULL values in the data and if we have predicted this values 
    instead of taking mean values, it would be better to pedict the TARGET variable.
''')

st.markdown('''
    #### Average of 500 Nearest Neighbor’s Target Value

    We have used the KNN Classifier to calculate this. First, we get 500 nearest neighbours for each data point using KNN Classifier 
    and take the probability output. It will give us the average of all target values if we keep the weights=’uniform’. 
    We have also used similar concepts for distance based probability values.
    Here we have created 2 new features “KNN_Uniform_500” and KNN_Distance_500.

    We have used the following features to get the probability output -
    - EXT_SOURCE_1 - original feature and imputed one
    - EXT_SOURCE_2 - original feature and imputed one
    - EXT_SOURCE_3 - original feature and imputed one
    - AMT_ANNUITY/AMT_CREDIT
    - DAYS_BIRTH
    - AMT_ANNUITY
    - AMT_CREDIT/AMT_GOODS_PRICE

    We have also tried various other K, like 750, but it is not improving the score in validation data. 
    So we decided that drop the feature based on 750 nearest neighbours.

''')

st.markdown('''
    #### Aggregation of Other Data

    There are several other data for each user in other files. We have to aggregate the data with other mail application_train/test data to get the best 
    performance. Initialy we have used average based aggregation. We haev seen that some features have very good feature importance. 
    We have tried similar aggregation concept using min and max function.
''')

st.markdown('''
    #### Weighted Average Credit Card Balance for Multiple Periods

    We have tried some weighted averages for credit card balances for a customer for the last few months. 
    Last month's credit card balance has more weightage than the previous one. Similarly, we are decreasing the weightage of the prior month. 
    We have applied this process for 3 different periods.

    - Last 12 Months weightage average of Credit Card Balance.
    - Last 24 Months weightage average of Credit Card Balance
    - Last 36 Months weightage average of Credit Card Balance

''')



