import modeling as md
from modeling.model_helper import custom_color_scale

import pandas as pd
import numpy as np
import math

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from math import ceil

class model_summarizer:
    """A visualization wrapper for the Model class

    Allows for the creation of quick/intuitive summary graphics to help aid in 
    extracting meaning from models.

    Currently supports analysis on a single mapper, no comparison functionality yet.

    """

    def __init__(self, model: md.Model):

        self.model = model
        self.zscores = self.define_zscore_df().groupby(['cluster_IDs']).mean().reset_index()




    def define_zscore_df(self):
        df_builder = pd.DataFrame()
        dfs = self.model.tupper.clean

        column_to_drop = [col for col in dfs.columns if dfs[col].nunique() == 1]
        dfs = dfs.drop(column_to_drop, axis=1)

        dfs['cluster_IDs']=(list(self.model.cluster_ids))

        #loop through all policy group dataframes
        for group in list(dfs['cluster_IDs'].unique()):
            zscore0 = pd.DataFrame()
            group0 = dfs[dfs['cluster_IDs']==group].drop(columns={'cluster_IDs'})
            #loop through all columns in a policy group dataframe
            for col in group0.columns:
                if col != "cluster_IDs":
                    mean = dfs[col].mean()
                    std = dfs[col].std()
                    zscore0[col] = group0[col].map(lambda x: (x-mean)/std)
            zscore0_temp = zscore0.copy()
            zscore0_temp['cluster_IDs'] = group
            df_builder = pd.concat([df_builder,zscore0_temp])
        return df_builder
    
    def get_group_identifiers(self):
        pg_identifiers = {key: [] for key in range(-1, int(self.zscores['cluster_IDs'].max())+1)}
        for column in self.zscores.columns:
            counter = -1
            for value in self.zscores[column]:
                if column!='cluster_IDs':
                    if abs(value) >= 1:
                        test = self.define_zscore_df()[self.define_zscore_df()['cluster_IDs']==counter]
                        if abs(test[column].std()/test[column].mean()) <= 1:
                            pg_identifiers[counter].append(column)
                counter+=1
        return pg_identifiers
    
    #########
    ## VIZ ##
    #########

    def show_PCA(self, colors=True):
        df=self.model.tupper.clean.copy()
        df['cluster_IDs'] = self.model.cluster_ids
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
                fig.add_trace(go.Scatter( x=plot['PC1'], y=plot['PC2'], mode='markers', name=cluster, marker=dict(color=custom_color_scale()[cluster_list.index(cluster)][1])))
        # plot with no colors
        else:
            fig = px.scatter(data_frame=pca_df, x='PC1', y='PC2', color_discrete_sequence=['grey'])

        fig.update_layout(template='simple_white', width=800, height=600)
        fig.show()

    def create_pie_charts(self):
        # Define the color map based on the dictionary values
        colors = []

        def reorder_colors(colors):
            n = len(colors)
            ordered = []
            for i in range(n):
                if i % 2 == 0:
                    ordered.append(colors[i // 2])
                else:
                    ordered.append(colors[n - (i // 2) - 1])
            return ordered

        for i in range(len(custom_color_scale()[:-3])):
            inst = custom_color_scale()[:-3]
            rgb_color = 'rgb' + str(tuple(int(inst[i][1][j:j+2], 16) for j in (1, 3, 5)))
            colors.append(rgb_color)
        
        colors = reorder_colors(colors)
        color_map = {key: colors[i % len(colors)] for i, key in enumerate(set.union(*[set(v['density'].keys()) for v in self.model.cluster_descriptions.values()]))}

        def get_subplot_specs(n):
            """
            Returns subplot specs based on the number of subplots.
            
            Parameters:
                n (int): number of subplots
                
            Returns:
                specs (list): 2D list of subplot specs
            """
            num_cols = min(3, n)
            num_rows = math.ceil(n / num_cols)
            specs = [[{"type": "pie"} for c in range(num_cols)] for r in range(num_rows)]
            return specs
    
        num_rows = math.ceil(len(self.model.cluster_descriptions) / 3)
        specs = get_subplot_specs(len(self.model.cluster_descriptions))

        dict_2 = {i: f'Group {i}' for i in range(len(self.model.cluster_descriptions))}
        dict_2 = {-1: 'Outliers', **dict_2}

        fig = make_subplots(rows=num_rows, 
                            cols=3, 
                            specs=specs, 
                            subplot_titles = [f"<b>{dict_2[key]}</b>: {self.model.cluster_descriptions[key]['size']} Members" for key in self.model.cluster_descriptions],
                            horizontal_spacing=0.1
        )
        
        for i, key in enumerate(self.model.cluster_descriptions):
            density = self.model.cluster_descriptions[key]['density']

            labels = list(density.keys())
            sizes = list(density.values())

            #labels_list = [labels.get(item, item) for item in labels]

            row = i // 3 + 1
            col = i % 3 + 1
            fig.add_trace(go.Pie(labels=labels, 
                                textinfo='percent',
                                values=sizes, 
                                textposition='outside',
                                marker_colors=[color_map[l] for l in labels], 
                                scalegroup=key,
                                hole=0.5,
                                ),
                        row=row, col=col)

        fig.update_layout(template='plotly_white', showlegend=True, height = 600, width = 800)

        fig.update_annotations(yshift=10)

        fig.update_traces( marker=dict(line=dict(color='white', width=3)))

        fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.22,
        xanchor="left",
        x=0
    ))

        # Show the subplot
        return fig
    
    def visualize_cc(self, col_list = []):

        df = self.model.tupper.raw
        df['cluster_IDs'] = self.model.cluster_ids
        
        if len(col_list)>0:
            col_list.append('cluster_IDs')
            df = df.loc[:, df.columns.isin(col_list)]

        fig = make_subplots(
                rows=math.ceil(len(df.columns.drop('cluster_IDs')) / 3), cols=3,
                horizontal_spacing=0.1,
                subplot_titles = df.columns.drop('cluster_IDs'))

        row=1
        col=1 

        dict_2 = {i: str(i) for i in range(len(list(df.cluster_IDs.unique())))}
        dict_2 = {-1: 'Outliers', **dict_2}

        for column in df.columns.drop('cluster_IDs'):
                for pg in list(dict_2.keys()):
                    fig.add_trace(go.Box(y = df[df['cluster_IDs']==pg][column], name=dict_2[pg], jitter=0.3, showlegend=False,
                    whiskerwidth=0.6, marker_size=3, line_width=1, boxmean=True,
                    marker=dict(color= custom_color_scale()[int(pg)][1])),
                    row=row, col=col)

                    try:
                        pd.to_numeric(df[column])
                        fig.add_hline(y=df[column].mean(), line_width=0.5, line_dash="dot", line_color="black", col=col, row=row, 
                        annotation_text='mean', annotation_font_color='gray', annotation_position="top right")
                    except ValueError:
                        ''

                col+=1
                if col == 4:
                    col=1
                    row+=1

        fig.update_layout(template='simple_white', height=(math.ceil(len(df.columns) / 3))*300, title_font_size=20)
        fig.show()