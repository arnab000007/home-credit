import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Home Credit Default Prediction",
    page_icon=":house:",
)


st.title('Home Credit Default Prediction')
st.markdown('''
    Many people are struggling to get loans from a financial institution due to insufficient Credit histories. 
    Because of this, people might have to take a loan from an untrustworthy lender to fulfil the purpose. 
    If they are getting a loan the interest rate is very high due to the risk of lack of credit history/credit score. 
    There is a very good chance to capture this market with a better approach. In this problem, we are also trying to 
    help Home Credit to have a better machine learning/Deep Learning approach to predict the probability of defaulting 
    of a borrower.
''')

st.subheader('Introduction')

st.markdown('''
    Home Credit(Organising Company) are trying to include the people in their lending channel who are neglected by the banking industry for a long time due to a lack of credit history. As a lending institution, they need to ensure to lend the Depositor's money to the lender with very few probabilities of default. To mitigate the default risk, the financial institutions were previously checking their Credit History or Credit Rating before lending the money. Some people have full potential to repay the loan, but because of a lack of credit history, they might not get the loan.
The company want to serve these underserved population using a variety of alternative data including telco and transactional information to predict their clients' repayment abilities. You can find some more details in the below link - [www.kaggle.com](https://www.kaggle.com/competitions/home-credit-default-risk/overview)
If they can use these data points efficiently then they can open their lending to a large population with minimal risks. From a borrower’s standpoint, they also borrow money for their needs from a trusty organization and do have some good experiences while taking loans.

''')

st.subheader('Business impact')

st.markdown('''
    The main business of a lending Institution is taking money from the depositor and lending the same money to a borrower to earn extra interest than the depositor’s interest. This extra interest is their revenue/profit. Sometimes the lender is taking the loan but is not able to repay the principal or interest amount. If borrowers are not able to repay the principal/interest then the institution will face difficulties to return money to the depositer.
In real life, there are some default cases which means some people are not able to repay the loan. Lending Institution is trying to reduce the number of default cases. To reduce this number, they are trying to improve their statistical model to predict the defaulting borrower based on the data available at our disposal. If they are successful then they can increase their revenue/profit with minimal risk.
''')

st.subheader('Contents')
st.markdown('- **_Data Analysis_**')