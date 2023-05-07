from model import Model
from model_helper import custom_color_scale

import pandas as pd
import numpy as np
import math

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

class Model_viz:
    """A visualization wrapper for the Model class
    """

    def __init__(self, model: Model, column_filter = ["ORISPL", "coal_FUELS", "NONcoal_FUELS", "ret_DATE", "PNAME", "FIPSST", "FIPSCNTY", "LAT", "LON", "Utility ID", "Entity Type"]):
        self._column_filter = column_filter
        self._color_mapping = {0:'#001219', 1:'#005f73',2:'#0a9396', 3:'#94d2bd', 4:'#e9d8a6', 5:'#ee9b00', 6:'#ca6702', 7:'#bb3e03', 8:'#ae2012', 9:'#9b2226', -1:'#a50026'}

        self.model = model
        self.aves = self.define_aves()
        self.zscore_df = self.define_zscore_df()
        self.policy_group_columns = self.define_policy_group_columns()
    
    @property
    def global_means(self):
        return self.data_cleaner(self.model.tupper.raw).mean()
    
    @property
    def global_stds(self):
        return self.data_cleaner(self.model.tupper.raw).std()
    
    @property
    def zscore_summary(self):
        return self.zscore_df.groupby(['cluster_IDs']).mean().reset_index()
    
    @property
    def encoding_info(self):
        print(self.print_info)

    @property
    def location_info(self):
        return self.aves.merge(self.model.tupper.raw[['LAT', 'LON', 'PNAME']], left_index=True, right_index=True, how='left')
    
    ####### functions to initialize the Model_viz class #######
    def define_policy_group_columns(self):
        policy_group_columns = {}
        for row_index in range(0, len(self.zscore_summary)):
            row = self.zscore_summary.iloc[row_index].drop('cluster_IDs')
            columns_to_include = []
            
            for i, value in row.iteritems():

                ### only add to policy_group_columns if cluster-mean z-score is >= 1
                if abs(value) >= 1:

                    ### only add to policy_group_columns if within 2 stds from global mean ###
                    if self.zscore_df[self.zscore_df['cluster_IDs']==self.zscore_summary['cluster_IDs'][row_index]][i].std()<=2:

                        columns_to_include.append(i)

            policy_group_columns[self.zscore_summary['cluster_IDs'][row_index]] = columns_to_include
        return policy_group_columns
    
    def define_aves(self):
        '''aves is a dataframe of all raw data columns with cluster_IDs'''
        aves = self.data_cleaner(self.model.tupper.raw).dropna()
        aves['cluster_IDs']=(list(self.model.cluster_ids))
        return aves

    def define_zscore_df(self):
        df_builder = pd.DataFrame()
        dfs = self.data_cleaner(self.model.tupper.raw).dropna()
        dfs['cluster_IDs']=(list(self.model.cluster_ids))
        #loop through all policy group dataframes
        for group in list(dfs['cluster_IDs'].unique()):
            zscore0 = pd.DataFrame()
            group0 = dfs[dfs['cluster_IDs']==group].drop(columns={'cluster_IDs'})
            #loop through all columns in a policy group dataframe
            for col in group0.columns:
                if col != "cluster_labels":
                    mean = self.global_means[col]
                    std = self.global_means[col]
                    zscore0[col] = group0[col].map(lambda x: (x-mean)/std)
            zscore0_temp = zscore0.copy()
            zscore0_temp['cluster_IDs'] = group
            df_builder = pd.concat([df_builder,zscore0_temp])
        return df_builder

    def data_cleaner(self, data):
        cleaned_data = data.drop(columns=self._column_filter)
        categorical_variables = cleaned_data.select_dtypes(
            exclude=["number"]
        ).columns.tolist()
        encoding_info = []  # Initialize list to accumulate column-wise information
        for column in categorical_variables:
            vals = cleaned_data[column].values
            cleaned_data[column] = self.integer_encoder(vals)
            encoding_info.append(f"{self.color.BOLD}{self.color.BLUE}{column}{self.color.END}: {np.unique(vals)}, {np.unique(self.integer_encoder(vals))}\n")
        self.print_info = (self.color.BOLD +self.color.UNDERLINE+ self.color.RED + "Column: Unique Values, Encoded Values\n\n" + self.color.END + '\n'.join(encoding_info))  # Create print_info variable
        return cleaned_data
    
    def integer_encoder(self, column_values):
        _, integer_encoding = np.unique(column_values, return_inverse=True)
        return integer_encoding
    
    class color:
        ## for coloring font and print outputs
        BLUE = '\033[94m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

    
    ####### Visualization Functions #######

    def visualize_cc(self,pgn:int, show_all=False):
        # column labels for showing boxplots of all columns
        if show_all:
            col_list = self.aves.columns.drop('cluster_IDs')
        else:
            col_list = self.policy_group_columns[pgn]

        fig = make_subplots(
                rows=math.ceil(len(col_list) / 3), cols=3,
                subplot_titles = col_list)

        row=1
        col=1 
        temp = list(self.aves['cluster_IDs'].unique())
        temp.sort()
        #plot
        for column in col_list:
            for pg in temp:
                fig.add_trace(go.Box(y = self.aves[self.aves['cluster_IDs']==pg][column], name = pg, jitter=0.3, showlegend=False, 
                whiskerwidth=0.6, marker_size=3, line_width=1, boxmean=True,
                marker=dict(color= custom_color_scale()[int(pg)][1])),
                row=row, col=col)
            # add mean line
            fig.add_hline(y=self.aves[column].mean(), line_width=0.5, line_dash="dot", line_color="black", col=col, row=row, 
            annotation_text='mean', annotation_font_color='gray', annotation_position="top right")
            # add median line
            fig.add_hline(y=self.aves[column].median(), line_width=0.75, line_dash="solid", line_color="grey", col=col, row=row, 
            annotation_text='median', annotation_font_color='gray', annotation_position="top right")

            col+=1
            if col == 4:
                col=1
                row+=1

        fig.update_layout(template='simple_white', height=(math.ceil(len(col_list) / 3))*300, title = f'Policy Group {pgn}', title_font_size=20)
        fig.show()

    def visualize_pg_scores(self):
        # pivot the df (makes plotting easier)
        zscore_group = pd.melt(self.zscore_summary, var_name='column', id_vars='cluster_IDs')
        # plot
        fig = px.strip(zscore_group, x='column', y='value', facet_col='cluster_IDs', color='cluster_IDs', color_discrete_map=self._color_mapping)
        fig.update_layout(template='plotly_white', yaxis_range = [-3,3])
        fig.update_xaxes(dict(tickfont = dict(size=4)))
        fig.show()

    def visualize_pg_scores_std(self):
        # group
        std = self.aves.groupby(['cluster_IDs']).std().reset_index()
        # calulate the coefficient of variation - used instead of std as it is standardized across variables
        std = abs(self.aves.groupby(['cluster_IDs']).std() / self.aves.groupby('cluster_IDs').mean()).reset_index()
        # pivot the df (makes plotting easier)
        std = pd.melt(std, var_name='column', id_vars='cluster_IDs')
        # plot
        fig = px.strip(std, x='column', y='value', facet_col='cluster_IDs', color='cluster_IDs', color_discrete_map=self._color_mapping)
        fig.add_hrect(y0=2, y1=std['value'].max(), line_width=0, fillcolor="red", opacity=0.1)
        fig.update_traces(marker = dict(size=5))
        fig.update_xaxes(dict(tickfont = dict(size=4)))
        fig.update_layout(template='plotly_white')
        fig.show()

    def show_PCA(self, colors=True):
        
        df=self.aves.copy()
        # Standardize the data
        X = StandardScaler().fit_transform(df)
        # Perform PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        # Create a DataFrame for the principal components
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        # Add the original cluster IDs to the PCA DataFrame
        pca_df['cluster_IDs'] = df['cluster_IDs'].values
        # plot with cluster-label based color scheme
        if colors:
            fig = go.Figure()
            cluster_list = list(pca_df['cluster_IDs'].unique())
            cluster_list.sort()
            for cluster in cluster_list:
                plot = pca_df[pca_df['cluster_IDs']==cluster]
                fig.add_trace(go.Scatter( x=plot['PC1'], y=plot['PC2'], mode='markers', name=cluster, marker=dict(color=self._color_mapping[cluster])))
        # plot with no colors
        else:
            fig = px.scatter(data_frame=pca_df, x='PC1', y='PC2', color_discrete_sequence=['grey'])

        fig.update_layout(template='simple_white', width=800, height=600)
        fig.show()

    def show_map(self, size='PLGENACL', min_size=5, max_size=15):
        locals = self.location_info.copy()

        def resize(df, column, min_value, max_value):
            # function to scale a df column to ensure that markers are sized properly regardlessly of the what they are being sized upon
            min_col = df[column].min()
            max_col = df[column].max()
            df[column] = ((df[column] - min_col) / (max_col - min_col)) * (max_value - min_value) + min_value
            return df
        
        locals=resize(locals, size, min_size, max_size)

        fig = go.Figure()
        cluster_list = list(locals['cluster_IDs'].unique())
        cluster_list.sort()
        for cluster in cluster_list:
            plot = locals[locals['cluster_IDs']==cluster]

            # log_plgenacl = np.log10(plot['PLGENACL'])

            # # normalize the logarithmic values to a range between 0 and 1 using min-max scaling
            # norm_log_plgenacl = (log_plgenacl - log_plgenacl.min()) / (log_plgenacl.max() - log_plgenacl.min())

            # # use the normalized values to set the marker size
            # marker_size = norm_log_plgenacl * 8

            fig.add_trace(go.Scattergeo(
                lat = plot['LAT'],
                lon=plot['LON'],
                text=plot["PNAME"],
                name=f'Policy Group {cluster}',

                mode="markers",
                marker_symbol='circle',
                marker = dict(
                    color=self._color_mapping[cluster], 
                    size=plot[size],
                    sizemode='diameter')
            ))
        fig.update_geos(dict(scope='usa',
                    showland = True,
                    landcolor = "rgb(247, 247, 247)",
                    subunitcolor = "rgb(100, 100, 100)",
                    showcountries=True,
                    countrycolor="black",
                    countrywidth=.1,
                    subunitwidth = 0.1),)

        fig['layout']['margin'].update(r=0,t=20,l=0,b=10)

        fig.update_annotations(font_size=12)
        fig.update_layout(template='simple_white', width = 800, height = 400,
        legend=dict(
            traceorder='grouped',  orientation="v", itemsizing='constant',
                itemwidth=40,),
        font = dict(size=10))
        #fig.update_traces(marker_line_width=)
        fig.show()


    def show_map2(self, size='PLGENACL', min_size=5, max_size=15, step_size=3):

        fig = make_subplots(rows=1, cols=2, column_widths=[0.05, 0.95],
                            specs = [[{"type": "scatter"},{"type": "scattergeo"}]],
                            horizontal_spacing=0.05,
                            subplot_titles=[f'Marker Size: {size}'])

        data = self.location_info[size]

        sizes = list(range(min_size, max_size + 1, step_size))

        # Get the minimum, maximum, and median values of the data
        data_min = round(np.min(data),0)
        data_max = round(np.max(data), 0)
        data_median = round(np.median(data),0)

        for i in range(len(sizes)-1):
            fig.add_trace(go.Scatter(
                x=[0],
                y=[i],
                mode='markers',
                marker=dict(size=sizes[i], sizemode='diameter', color='grey'),
                showlegend=False
            ),row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[1],
                y=[i],
                mode='text',
                text=[str(sizes[i])],
                showlegend=False,
                textfont=dict(size=16, color='black'),
                textposition='middle right'
            ),row=1, col=1)
            fig.update_layout(
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-1,1]
            ),
            yaxis=dict(
                tickvals=list(range(len(sizes))),
                ticktext=[data_min, data_median, data_max],  # <-- Use data min/max/median to label y-axis ticks
                showgrid=False,
                zeroline=False,
                tickmode='array'
            ),
            #showlegend=False,

            template='none'
        )

        locals = self.location_info.copy()

        cluster_list = list(locals['cluster_IDs'].unique())
        cluster_list.sort()

        def resize(df, column, min_value, max_value):
            min_col = df[column].min()
            max_col = df[column].max()
            df[column] = ((df[column] - min_col) / (max_col - min_col)) * (max_value - min_value) + min_value
            return df
        locals=resize(locals, size, min_size, max_size)

        for cluster in cluster_list:
            plot = locals[locals['cluster_IDs']==cluster]

            # log_plgenacl = np.log10(plot['PLGENACL'])

            # # normalize the logarithmic values to a range between 0 and 1 using min-max scaling
            # norm_log_plgenacl = (log_plgenacl - log_plgenacl.min()) / (log_plgenacl.max() - log_plgenacl.min())

            # # use the normalized values to set the marker size
            # marker_size = norm_log_plgenacl * 8

            fig.add_trace(go.Scattergeo(
                lat = plot['LAT'],
                lon=plot['LON'],
                text=plot["PNAME"],
                name=f'Policy Group {cluster}',

                mode="markers",
                marker_symbol='circle',
                marker = dict(
                    color=self._color_mapping[cluster], 
                    size=plot[size],
                    sizemode='diameter')
            ), row=1, col=2)
        fig.update_geos(dict(scope='usa',
                    showland = True,
                    landcolor = "rgb(247, 247, 247)",
                    subunitcolor = "rgb(100, 100, 100)",
                    showcountries=True,
                    countrycolor="black",
                    countrywidth=.1,
                    subunitwidth = 0.1),)

        fig['layout']['margin'].update(r=0,t=60,l=60,b=40)

        fig.update_annotations(font_size=12)
        fig.update_layout(template='none', 
        legend=dict(
            traceorder='grouped',  itemsizing='constant',
                itemwidth=40,),
        font = dict(size=10))
        #fig.update_traces(marker_line_width=)

        fig.show()