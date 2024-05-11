# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:45:30 2024

@author: WILLIAM LEE
"""

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from io import BytesIO
#import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.cluster import DBSCAN
def load_data():
    return pd.read_csv("tfidf_features_reduced.csv")

data = load_data()
def tryy():
    dbscan = DBSCAN(eps=0.6, min_samples=6)  # Adjust parameters as needed
    predicted_clusters_dbscan = dbscan.fit_predict(data)
    return predicted_clusters_dbscan

predicted_clusters_dbscan = tryy()
fig, ax = plt.subplots(figsize=(8, 6)) 
dat = data.values
for cluster_label in np.unique(predicted_clusters_dbscan):
    if cluster_label == -1:
        # Points labeled as noise (cluster label -1)
        noise_points = dat[predicted_clusters_dbscan == -1]
        ax.scatter(noise_points[:, 0], noise_points[:, 1], label='Noise', color='gray', alpha=0.5)
    else:
        # Points belonging to a cluster
        cluster_points = dat[predicted_clusters_dbscan == cluster_label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}', cmap='viridis')
ax.set_title(f'DBSCAN Clustering with {len(np.unique(predicted_clusters_dbscan))} Clusters')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()
st.pyplot(fig)