import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


class PlotingUtility:

    def __init__(self, df):
        """

        :param df: It is the main dataset in which we will plot the data
        """
        self.df = df
        self.title = "Count of column {} for each values"
        self.title_hue = "Percentage of each {} values with {} values"
        self.colours = sns.color_palette()
        plt.rcParams.update({'font.size': 22})

    def count_plot_each_category(self, column, hue, fig_size=(16, 8)):
        fig, ax = plt.subplots(figsize=fig_size)
        ax = sns.countplot(x=column, hue=hue, data=self.df, ax=ax)
        plt.title('Count plot of {} for each {} values'.format(column, hue), fontsize=20)
        # ax.yaxis.set_tick_params(labelsize = 20)
        #
        return fig

    def percentage_plot_each_category(self, column, hue='', fig_size=(16, 8)):
        """

        :param column:
        :param hue:
        :param fig_size:
        :return: Seaborn Figure
        """
        if hue == '':
            fig, ax = plt.subplots(figsize=fig_size)
            sns.countplot(x=column, data=self.df, ax=ax)
            plt.title(self.title.format(column), fontsize=20)
            return fig
        else:
            df1 = self.df.groupby(column)[hue].value_counts(normalize=True)
            df1 = df1.mul(100)
            df1 = df1.rename('Percent').reset_index()

            g = sns.catplot(x=column, y='Percent', hue=hue, kind='bar', data=df1, height=fig_size[1], aspect=2)
            g.ax.set_ylim(0, 100)
            plt.title(self.title_hue.format(column, hue), fontsize=20)
            for p in g.ax.patches:
                txt = str(p.get_height().round(2)) + '%'
                txt_x = p.get_x()
                txt_y = p.get_height()
                g.ax.text(txt_x, txt_y, txt, fontsize=20)

            myTable = PrettyTable([column, "Percentage Of Total", "#Positive Records", "#Negative Records"])
            df_count = self.df[column].value_counts()
            lst_column = [hue, column]
            df_counts = self.df[lst_column].value_counts()
            for i in list(df_count.index):
                myTable.add_row([
                    i,
                    round(df_count[i] * 100.0 / self.df.shape[0], 2),
                    df_counts[(1, i)] if (1, i) in df_counts.index else 0,
                    df_counts[(0, i)] if (0, i) in df_counts.index else 0
                ])

            return g, myTable

    def plot_histogram(self, column, min_percentile=0.0, max_percentile=100, log_scale=True, kde=True):
        df_fill = self.df.copy()
        df_fill = df_fill[df_fill[column].notnull()]

        min_max = np.percentile(df_fill[column].values, [min_percentile, max_percentile])
        if log_scale:
            df_fill = df_fill[df_fill[column] != 0.00]
        if min_percentile != 0.00:
            df_fill = df_fill[df_fill[column] >= min_max[0]]

        if max_percentile != 100.0:
            df_fill = df_fill[df_fill[column] <= min_max[1]]

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        sns.histplot(x=column, stat='probability', data=df_fill[df_fill['TARGET'] == 1], bins=20,
                     kde=kde, log_scale=log_scale, legend=True, ax=ax, label='Class 1', color=self.colours[0])
        sns.histplot(x=column, stat='probability', data=df_fill[df_fill['TARGET'] == 0], bins=20,
                     kde=kde, log_scale=log_scale, legend=True, ax=ax, color=self.colours[1], label='Class 0')
        plt.legend()
        plt.title('Histogram of {} for each Target value'.format(column), fontsize=20)
        return fig

    def plot_cdf(self, column, log_scale):

        fg = sns.displot(data=self.df, x=column, hue="TARGET", kind="ecdf", height=8, aspect=2, log_scale=log_scale)
        plt.title("CDF of {}".format(column), fontsize=20)
        return fg

    def print_percentile(self, column, per_arr):
        p_tbl = PrettyTable(["Percentile 1", "Value 1", " ", "Percentile 2", "Value 2"])
        per_arr = [i for i in list(np.round(per_arr, 2)) if i <= 100.00]
        for i in range(int(np.ceil(len(per_arr) / 2.0))):
            r = [per_arr[i * 2], np.round(np.percentile(self.df[column], per_arr[i * 2]), 2), ' ', ]
            if i * 2 + 1 >= len(per_arr):
                r.extend(['', ''])
            else:
                r.extend([per_arr[i * 2 + 1], np.round(np.percentile(self.df[column], per_arr[i * 2 + 1]), 2)])
            p_tbl.add_row(r)
        return p_tbl

    def get_null_count(self, column):
        return self.df[column].isnull().sum()

    def get_null_distribution(self, column):
        df_fill = self.df.copy()
        df_fill = df_fill[df_fill[column].isnull()]

        my_tbl = PrettyTable(['TARGET', "% Of Total", "Records Count", "% of Null Records"])
        pos_rows = sum(df_fill['TARGET'] == 1)
        neg_rows = sum(df_fill['TARGET'] == 0)

        my_tbl.add_row([0, 100.0 * neg_rows / self.df.shape[0], neg_rows, 100.0 * neg_rows / df_fill.shape[0]])
        my_tbl.add_row([1, 100.0 * pos_rows / self.df.shape[0], pos_rows, 100.0 * pos_rows / df_fill.shape[0]])
        return my_tbl

    def plot_multivariate_categorical(self, col_name, col1, col2, operation, hue='TARGET',fig_size=(16,8), null_imputation='mean'):
        df = self.df.copy();
        df[col_name] = self.operate(df, col1, col2, operation)

        df1 = df.groupby(col_name)[hue].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('Percent').reset_index()

        g = sns.catplot(x=col_name, y='Percent', hue=hue, kind='bar', data=df1, height=fig_size[1], aspect=2)
        g.ax.set_ylim(0, 100)
        plt.title(self.title_hue.format(col_name, hue), fontsize=20)
        for p in g.ax.patches:
            txt = str(p.get_height().round(2)) + '%'
            txt_x = p.get_x()
            txt_y = p.get_height()
            g.ax.text(txt_x, txt_y, txt, fontsize=20)

        myTable = PrettyTable([col_name, "Percentage Of Total", "#Positive Records", "#Negative Records"])
        df_count = df[col_name].value_counts()
        lst_column = [hue, col_name]
        df_counts = df[lst_column].value_counts()
        for i in list(df_count.index):
            myTable.add_row([
                i,
                round(df_count[i] * 100.0 / df.shape[0], 2),
                df_counts[(1, i)] if (1, i) in df_counts.index else 0,
                df_counts[(0, i)] if (0, i) in df_counts.index else 0
            ])

        return g, myTable

    def operate(self, df, col1, col2, operation):

        if operation == 'Greater Than':
            return df['AMT_INCOME_TOTAL'] > df['AMT_CREDIT']
        elif operation == 'Divide':
            return df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
