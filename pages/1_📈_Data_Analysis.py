import streamlit as st
import pandas as pd
import numpy as np
from Utility.PlotingUtility import PlotingUtility
from Utility.MultiVariateUtility import MultiVariateUtility
import seaborn as sns

st.set_page_config(
    page_title="Data Analysis",
    page_icon=":chart_with_upwards_trend:"
)

ALL_ROWS_COUNT = 307511
CUR_RECORDS_COUNT = 307511


@st.cache(allow_output_mutation=True)
def load_data(num_rows):
    _data = pd.read_csv('Data/application_train.csv', nrows=num_rows)
    _data_obs = pd.read_csv('Data/EDA_Observations.csv')
    _col_type = pd.read_csv('Data/column_type.csv')
    _cat_cols = list(_col_type[_col_type['Type'] == 'Categorical']['Col'].values)
    _num_cols = list(_col_type[_col_type['Type'] == 'Numerical']['Col'].values)
    return _data, _cat_cols, _num_cols, num_rows


@st.cache(allow_output_mutation=True)
def load_multivariate():
    _data = pd.read_csv('Data/multivariate.csv')
    return _data


obs_mark = """
    #### Observations
    """

# def observation_text(col):
#     text = ""
#     for i, v in enumerate(data_obs[data_obs['COL'] == col]['Observations'].values):
#         text = text + '{}. {} \n'.format(i + 1, v)
#     return text


st.title('Home Credit EDA')

# Side Menu
st.sidebar.header("Exploratory Data Analysis")
percent_data = st.sidebar.slider(
    'Filter the data',
    0, 100, 25, key="percent_rows")

# Loading the main data
data, cat_cols, num_cols, CUR_RECORDS_COUNT = load_data(np.floor(ALL_ROWS_COUNT * percent_data / 100))

st.sidebar.subheader("Target Value")
plot_util = PlotingUtility(data)
if st.sidebar.checkbox("Display Target Value Count", value=True):
    st.subheader('Target Values Distribution')
    fig = plot_util.percentage_plot_each_category(column='TARGET')
    st.pyplot(fig)
    # st.markdown(obs_mark)
    # st.write(observation_text('TARGET'))

type_analysis = st.sidebar.radio(
    "Select the analysis type",
    ('Single Variate', 'Multi Variate'), index=0)

if type_analysis == 'Single Variate':

    st.sidebar.markdown('## Single Variate Analysis')
    st.sidebar.subheader("Categorical Features")
    opt_categorical = st.sidebar.multiselect(
        'Please select the features to display Chart',
        cat_cols,
        [cat_cols[0]]
    )
    if len(opt_categorical) > 0:
        st.subheader('Categorical Features Distributions')

    for sel_col in opt_categorical:
        with st.spinner('Wait for it...'):
            st.markdown("#### Distribution of {}".format(sel_col))
            fig, tbl = plot_util.percentage_plot_each_category(column=sel_col, hue='TARGET')

            st.pyplot(fig)
            show_count_plot = st.checkbox("Show Count Plot", value=False, key='show_count_plot_' + sel_col)
            if show_count_plot:
                with st.spinner('Wait for it...'):
                    st.pyplot(plot_util.count_plot_each_category(sel_col, 'TARGET'))

            st.write(tbl)
            # st.markdown(obs_mark)
            # st.write(observation_text(sel_col))

            # Getting the rows with Null value
            null_count = plot_util.get_null_count(sel_col)
            if null_count > 0:
                st.markdown("**Number of null value for this column : {} out of {}, Percentage: {}**".format(
                    null_count, CUR_RECORDS_COUNT, np.round(100.0 * null_count / CUR_RECORDS_COUNT, 2)))
                show_null_cat = st.checkbox("Show Null Values count", value=False, key='show_null_cat_' + sel_col)
                if show_null_cat:
                    st.write(plot_util.get_null_distribution(sel_col))

    st.sidebar.subheader("Numerical Features")
    opt_numerical = st.sidebar.multiselect(
        'Please select the features to display Chart',
        num_cols,
        []
    )

    if len(opt_numerical) > 0:
        st.subheader('')
        st.subheader('Numerical Features Distributions')

    for sel_col in opt_numerical:
        with st.spinner('Wait for it...'):
            st.markdown("#### Distribution of {}".format(sel_col))

            min_max_percentile = st.slider(
                'Filter the data: Range of Percentile',
                0.0, 100.0, (0.0, 100.0), key='range_selector' + sel_col
            )

            log_scale = st.checkbox("Log Scale", value=False, key='log_scale_' + sel_col)
            kde = st.checkbox("Compute a kernel density estimation", value=True, key='kde_' + sel_col)
            fig = plot_util.plot_histogram(sel_col, min_percentile=min_max_percentile[0],
                                           max_percentile=min_max_percentile[1], log_scale=log_scale,
                                           kde=kde)
            st.pyplot(fig)

            show_cdf = st.checkbox("Show CDF", value=False, key='show_cdf_' + sel_col)
            if show_cdf:
                with st.spinner('Wait for it...'):
                    st.pyplot(plot_util.plot_cdf(sel_col, log_scale=log_scale))

            show_percentile = st.checkbox("Show Percentile", value=False, key='show_percentile_' + sel_col)
            if show_percentile:
                with st.spinner('Wait for it...'):
                    percentile_min_max = st.slider('Filter the data: Range of Percentile',
                                                   0.0, 100.0, (0.0, 100.0), key='percentile_slider_' + sel_col,
                                                   step=0.1)
                    step_size = (percentile_min_max[1] - percentile_min_max[0]) / 10.0
                    percentile_arr = np.arange(percentile_min_max[0], percentile_min_max[1] + step_size, step_size) \
                        if step_size != 0.00 else np.array([percentile_min_max[1]])

                    st.write(plot_util.print_percentile(sel_col, percentile_arr))

            null_count = plot_util.get_null_count(sel_col)
            if null_count > 0:
                st.markdown("**Number of null value for this column : {} out of {}, Percentage: {}**".format(
                    null_count, CUR_RECORDS_COUNT, np.round(100.0 * null_count / CUR_RECORDS_COUNT, 2)))
                show_null = st.checkbox("Show Null Values count", value=False, key='show_null_' + sel_col)
                if show_null:
                    st.write(plot_util.get_null_distribution(sel_col))

        # st.markdown(obs_mark)
        # st.write(observation_text(sel_col))
elif type_analysis == 'Multi Variate':
    st.sidebar.markdown('## Multivariate Analysis')
    mul_cols = load_multivariate()

    mul_option = st.sidebar.selectbox(
        'Please select a Feature', mul_cols['Col'].values
        , 0)

    curr_rows = mul_cols[mul_cols['Col'] == mul_option]
    print(curr_rows)
    mul_variate_plt_util = MultiVariateUtility(data, mul_option,
                                               curr_rows['Col1'].iloc[0],
                                               curr_rows['Col2'].iloc[0],
                                               curr_rows['Operation'].iloc[0])
    if curr_rows['ploy_type'].iloc[0] == 'Categorical':
        with st.spinner('Wait for it...'):
            st.markdown("#### Distribution of {}".format(mul_option))

            fig, tbl = mul_variate_plt_util.plot_multivariate_categorical()

            st.pyplot(fig)
            st.write(tbl)
            show_count_plot = st.checkbox("Show Count Plot", value=False, key='show_count_plot_' + mul_option)
            if show_count_plot:
                with st.spinner('Wait for it...'):
                    st.pyplot(mul_variate_plt_util.count_plot_each_category(mul_option, 'TARGET'))
    elif curr_rows['ploy_type'].iloc[0] == 'Numerical':
        with st.spinner('Wait for it...'):
            st.markdown("#### Distribution of {}".format(mul_option))

            min_max_percentile = st.slider(
                'Filter the data: Range of Percentile',
                0.0, 100.0, (0.0, 100.0), key='range_selector' + mul_option
            )

            log_scale = st.checkbox("Log Scale", value=False, key='log_scale_' + mul_option)
            kde = st.checkbox("Compute a kernel density estimation", value=True, key='kde_' + mul_option)

            fig = mul_variate_plt_util.plot_histogram(mul_option, min_percentile=min_max_percentile[0],
                                                      max_percentile=min_max_percentile[1], log_scale=log_scale,
                                                      kde=kde)
            st.pyplot(fig)

            show_cdf = st.checkbox("Show CDF", value=False, key='show_cdf_' + mul_option)
            if show_cdf:
                with st.spinner('Wait for it...'):
                    st.pyplot(mul_variate_plt_util.plot_cdf(mul_option, log_scale=log_scale))

            show_percentile = st.checkbox("Show Percentile", value=False, key='show_percentile_' + mul_option)
            if show_percentile:
                with st.spinner('Wait for it...'):
                    percentile_min_max = st.slider('Filter the data: Range of Percentile',
                                                   0.0, 100.0, (0.0, 100.0), key='percentile_slider_' + mul_option,
                                                   step=0.1)
                    step_size = (percentile_min_max[1] - percentile_min_max[0]) / 10.0
                    percentile_arr = np.arange(percentile_min_max[0], percentile_min_max[1] + step_size, step_size) \
                        if step_size != 0.00 else np.array([percentile_min_max[1]])

                    st.write(mul_variate_plt_util.print_percentile(mul_option, percentile_arr))


