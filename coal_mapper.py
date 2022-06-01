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



def egrid_clean(csv_path:str, eyear:str):
    '''This function takes in a ~specific coal data set and performs the following cleaning:
    - Sorts out non-coal fired powerplants
    - changes na to 0
    - Selects relevant pollution data
    - sorts out US territories
    - makes labels with the name, state, and year of data
    Returns a pandas data frame with the cleaned data.
'''
    df = pd.read_csv(csv_path)
    df.fillna(0)
    df2 = df[df['COALFLAG'] == 1]
    if (df2.size <2):
        df2 = df[df['COALFLAG'] == 'Yes']
    df = df2
    df['Year'] = eyear
    df = df[['Year', 'ORISPL', 'PSTATABB', 'LAT', 'LON', 'PNAME', 'FIPSST', 'FIPSCNTY',  'CAPFAC', 'PLNGENAN', 'PLNOXAN', 'PLSO2AN', 'PLCO2AN']]
    df = df.assign(label=eyear + ": " + df['PNAME'] + ", " + df['PSTATABB'])
    df.reset_index(drop=True, inplace=True)
    return df


def add_ccc (egrid:str, ccc:str):
    '''combines egrid and coal cost crossover datasets to add economic data and renewable replacement factor:
    - creates bin variable: 
        bin = 0 means that coal is teh cheapest resource
        bin = 1 means that either solar or wind is cheaper than current coal, according to CCC methods

    returns a smaller dataset: 'Total Coal Going-Forward Cost', 'FIPSST', 'ID', 'CAPFAC', 'PLCO2AN', 'PLNGENAN', 'PLNOXAN', 'PLSO2AN', 'bin', 'label', '% least cost resource less than coal', LCR
    '''

    ccc = ccc.rename(columns={"EIA Plant ID": "ID"})
    egrid = egrid.rename(columns={"ORISPL": "ID"})

    egrid = egrid.astype(float, errors = "ignore")
    ccc['ID'] = ccc['ID'].astype(float)

    mergedf = ccc.merge(egrid, how = "inner", left_on='ID',right_on='ID')
    mergedf = mergedf.rename(columns={"Overall least cost resource": "LCR"})

    mergedf["bin"] = ""

    for i, row in mergedf.iterrows():
        if (mergedf['LCR'].at[i] == 'Coal'):
            mergedf.at[i,'bin'] = 0
        else :
            mergedf.at[i,'bin'] = 1
    
    small = mergedf[['label', 'FIPSST', 'ID', 'CAPFAC', 'PLCO2AN', 'PLNGENAN', 'PLNOXAN', 'PLSO2AN', 'bin', 'LCR', 'Total Coal Going-Forward Cost', '% least cost resource less than coal']]
    
    return small
    

def add_yalePO (egridCCC:str, yale:str):
    '''ADDS PO, the percent that oppose setting CO2 Limits for coal powerplants in that state'''

    states = egridCCC
    states['PO'] = ""
    yale = yale[['GEOID', 'CO2limits','CO2limitsOppose']]
    yale = yale.truncate(before=0, after=53)
    states.head()   

    for i, row in yale.iterrows():
        for y, row in states.iterrows():
            if (states['FIPSST'].at[y] == yale['GEOID'].at[i]):
                states.PO[y] = yale.CO2limitsOppose[i]
    
    states['PO'] = states['PO'].astype(float)
    states['bin'] = states['bin'].astype(float)
    return states

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

def make_mapper(cubes:int, clusters:int, overlap:int, fulldf:pd.DataFrame, labelsdf:pd.DataFrame, fileP:str,):
    
    '''This function takes in a dataframe set, data labels, and colors to make a mapper object with the specified number of groups, covers, and percent overlap.
    Returns/updates an HTML based mapper graph
    '''
    #map with 2 lenses

    df2 = fulldf
    df = labelsdf

    colors = df2
    labels = mapper_labels(df['label'])


    mapper = km.KeplerMapper(verbose=2)

    projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
    projector.fit(df2)
    lens1 = projector.decision_function(df2)
    lens2 = mapper.fit_transform(df2, projection="knn_distance_5")


    lens = np.c_[lens1, lens2]

    graph_new = mapper.map(
        lens,
        df2,
        remove_duplicate_nodes=True,
        cover=km.Cover(n_cubes=cubes, perc_overlap=overlap),
        clusterer=cluster.AgglomerativeClustering(clusters))
        #clusterer = sklearn.cluster.MiniBatchKMeans(n_clusters=clusters, random_state=1618033))
        #clusterer = sklearn.cluster.KMeans(n_clusters=clusters, random_state=1618033))
    
    my_colorscale = [[0.0, '#001219'],
             [0.1, '#005f73'],
             [0.2, '#0a9396'],
             [0.3, '#94d2bd'],
             [0.4, '#e9d8a6'],
             [0.5, '#ee9b00'],
             [0.6, '#ca6702'],
             [0.7, '#bb3e03'],
             [0.8, '#ae2012'],
             [0.9, '#9b2226'],
             [1.0, '#a50026']]

    mapper.visualize(
        graph_new,
        path_html=fileP,
        title="Coal Trial 1 (" +str(cubes)+ " cube(s) at " +str(overlap*100) + "% overlap) with Plant Labels",
        custom_tooltips=labels,
        color_values = colors,
        colorscale=my_colorscale,
        color_function_name=list(df2.columns),
        node_color_function=['mean', 'median', 'max'],
        include_searchbar = True)

    return graph_new

    

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
    colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

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