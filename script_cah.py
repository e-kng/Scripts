# CLASSIFICATION ASCENDANTE HIERARCHIQUE

#########################################################################################################

# Variables à définir
    # Import et traitement des données
path = '' # dossier contenant les données
file = '' # nom du fichier contenant les données
sep = '\t' # séparateur de colonnes
index = 0 # nom ou numero de colonne
drop_col = [] # nom des colonnes à ignorer

    # Paramètres de prétraitement des données avant CAH
on_acp_data = 1 # CAH après réduction de dimension par ACP
nb_components = 2 
on_kmeans = 0 # Prégroupement des données dans le cas de larges datasets
nb_clust = 20 # nombre de clusters pour kmeans

# Affichage des figures (1 pour oui, 0 pour non)
dendrogram_plot = 1 
display_names = 0 # affichage des noms des observations

# Extraction des résultats des clusters
nb_clusters = 5 # nombre de clusters pour le découpage

# Export des données 
export_data = 0

#########################################################################################################

# import des librairies
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, set_link_color_palette
from scipy.cluster.hierarchy import dendrogram 
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
raw_data = pd.read_csv(path+file, sep=sep, index_col=index)

# Préparation des données pour la CAH
data = raw_data.drop(drop_col, axis=1)
data = data.fillna(raw_data.mean()) # remplacement des valeurs manquantes par la moyenne de la variable
X = data.values

# Centrage et réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Réduction des dimensions par ACP
if on_acp_data == 1:
    pca = decomposition.PCA(n_components=nb_components)
    pca.fit(X_scaled)
    X_scaled = pca.transform(X_scaled)

# Prégroupement par K-means
if on_kmeans == 1:
    km = KMeans(n_clusters = nb_clust)
    km.fit(X_scaled)
    X_scaled = km.cluster_centers_

# CAH
Z = linkage(X_scaled, 'ward')

# GRAPHIQUES
plt.style.use('seaborn-whitegrid')
fontsize_axes = 12
fontsize_ticks = 10
fontsize_title = 14

def display_dendrogram(Z):
    plt.figure(figsize=(25,10))

    # colors
    pal = sns.color_palette('cubehelix')
    pal = pal.as_hex()
    set_link_color_palette(list(pal))

    if display_names == 1:
        labels = raw_data.index
    else:
        labels = range(len(X_scaled))

    dendrogram(Z, labels = labels, 
               orientation = 'top',  above_threshold_color='grey')
    
    # legends
    plt.xticks(fontsize=fontsize_ticks)
    plt.ylabel("Distance", fontsize=fontsize_axes)
    plt.yticks(fontsize=fontsize_ticks)
    plt.title("Hierarchical Clustering Dendrogram", fontsize=fontsize_title, loc='left')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.show()

if dendrogram_plot == 1:
    display_dendrogram(Z)

# Découpage des clusters
clusters = fcluster(Z, nb_clusters, criterion='maxclust')

# Concaténation des clusters aux données
data_with_clusters = raw_data.copy()

if on_kmeans == 0:
    # dans le cas d'une CAH seule
    data_with_clusters['clusters'] = clusters 
else:
    # dans le cas d'une CAH sur K-means
    data_with_clusters['clusters'] = km.labels_
    map_clusters = {x:y for x,y in zip(range(20), clusters)}
    data_with_clusters['clusters'] = data_with_clusters['clusters'].map(map_clusters)

# Export des données
if export_data == 1:
    data_with_clusters.to_pickle(path+file.split('.')[0]+'_clusters.pickle')
