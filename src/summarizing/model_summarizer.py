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

from visualization_helper import _define_zscore_df

class model_summarizer:
    """A visualization wrapper for the Model class

    Allows for the creation of quick/intuitive summary graphics to help aid in 
    extracting meaning from models.

    Currently supports analysis on a single mapper, no comparison functionality yet.

    """

    def __init__(self, model: md.Model):

        self.model = model
        self.zscores = _define_zscore_df(self.model).groupby(['cluster_IDs']).mean().reset_index()

    
    def get_group_identifiers(self):
        pg_identifiers = {key: [] for key in range(-1, int(self.zscores['cluster_IDs'].max())+1)}
        for column in self.zscores.columns:
            counter = -1
            for value in self.zscores[column]:
                if column!='cluster_IDs':
                    if abs(value) >= 1:
                        test = _define_zscore_df(self.model)[_define_zscore_df(self.model)['cluster_IDs']==counter]
                        if abs(test[column].std()/test[column].mean()) <= 1:
                            pg_identifiers[counter].append(column)
                counter+=1
        return pg_identifiers
    
    #########
    ## VIZ ##
    #########

    def visualize_PCA(self, colors=True):
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

    