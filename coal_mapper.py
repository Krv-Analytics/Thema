#class coal_mapper:

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt


import kmapper as km
from sklearn import ensemble
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import time
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn import ensemble, cluster

import numpy as np
import plotly.graph_objects as go

from sklearn.cluster import AgglomerativeClustering

#### IMPORTANT FUNCTIONS
def ORISPLmerge(df1, df2):
    return df1.merge(df2, how='left', left_on='ORISPL', right_on='ORISPL')

######
#### OTHER FUNCTIONS
######

def coal_PCA (df:str):
    '''
    Graphs a Principle component analysis with 2 principle components
    df can only consist of numbers, no words/labels. Ensure you supply a subset of the dataset that conforms to this restriction'''
    x = df
    y = StandardScaler().fit_transform(x)

    # apply PCA
    pca = PCA(n_components=2)
    X = pca.fit_transform(y)

    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=list(x.columns))
    loadings

    def loading_plot(coeff, labels):
        n = coeff.shape[0]
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1], head_width = 0.05, head_length = 0.05, color = '#21918C',alpha = 0.5)
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = '#21918C', ha = 'center', va = 'center')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid()

    fig, ax = plt.subplots(figsize = (7,7))
    loading_plot(pca.components_.T, list(x.columns))


def mapper_labels (df:pd.DataFrame):

    '''This function takes in dataframe to use as the data labels for the mapper.
    Returns a dataframe
    '''

    names = df.to_numpy()
    return names

def view_kmeans(df:str, numclusters:int):

# data

    X =StandardScaler().fit_transform(df)

# #############################################################################
# Clustering with KMeans

    k_means = KMeans(init="k-means++", n_clusters=numclusters, n_init=10)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0

# #############################################################################
# Clustering with MiniBatchKMeans

    mbk = MiniBatchKMeans(
        init="k-means++",
        n_clusters=numclusters,
        batch_size=numclusters,
        n_init=10,
        max_no_improvement=10,
        verbose=0,
    )
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0

# #############################################################################
# Plot result

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ["#4EACC5", "#FF9C34", "#4E9A06", "#3082be"]

# COLOR MiniBatchKMeans and the KMeans algorithm.
    k_means_cluster_centers = k_means.cluster_centers_
    order = pairwise_distances_argmin(k_means.cluster_centers_, mbk.cluster_centers_)
    mbk_means_cluster_centers = mbk.cluster_centers_[order]

    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

# KMeans
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(numclusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("KMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_batch, k_means.inertia_))

# MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(numclusters), colors):
        my_members = mbk_means_labels == k
        cluster_center = mbk_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("MiniBatchKMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_mini_batch, mbk.inertia_))

# Initialise the different array to all False
    different = mbk_means_labels == 4
    ax = fig.add_subplot(1, 3, 3)

    for k in range(numclusters):
        different += (k_means_labels == k) != (mbk_means_labels == k)

    identic = np.logical_not(different)
    ax.plot(X[identic, 0], X[identic, 1], "w", markerfacecolor="#bbbbbb", marker=".")
    ax.plot(X[different, 0], X[different, 1], "w", markerfacecolor="m", marker=".")
    ax.set_title("Difference")
    ax.set_xticks(())
    ax.set_yticks(())

    plt.show()

def percentCO2reduxMap(mapdf:pd.DataFrame, all:pd.DataFrame):

    def make_chloropheth(subset:pd.DataFrame, all:pd.DataFrame):
        state_total = all[all['decom'] == 0].groupby('PSTATABB')['PLCO2AN'].sum().reset_index()
        state_reduction = subset[subset['decom'] == 0].groupby('PSTATABB')['PLCO2AN'].sum().reset_index()
        state_total = state_total.merge(state_reduction, how='left', on='PSTATABB').fillna(0)
        state_total['percentRedux'] = state_total['PLCO2AN_y']/state_total['PLCO2AN_x']
        return state_total
        
    state_total = make_chloropheth(mapdf, all)

    num = len(mapdf)
    redux = int(100*(mapdf['PLCO2AN'].sum() / all[all['decom'] == 0]['PLCO2AN'].sum()))

    fig = go.Figure(data=[go.Choropleth(locations=state_total['PSTATABB'], # Spatial coordinates
                                        z = state_total['percentRedux'].astype(float), # Data to be color-coded
                                        locationmode = 'USA-states', # set of locations match entries in `locations`
                                        colorscale = 'earth',
                                        colorbar_title = "% State Reduction"),
            go.Scattergeo(
                lon = mapdf['LON'],
                lat = mapdf['LAT'],
                text = mapdf['label'],
                marker=dict(color=list(range(0)),
                    colorscale='viridis',
                    size=mapdf['PLNGENAN']/400000),
                #marker_color = df['CO2limits'],
            )])

    fig.update_layout(
        
        title_text = f'{num} Plant Closures - {redux}% CO2 Reduction Nationwide',
        geo_scope='usa', # limite map scope to USA
    )

    fig.show()

