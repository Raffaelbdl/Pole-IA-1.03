from utils.merging import ajouter_actions
from random import shuffle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from math import inf
from random import randint
from progress.bar import IncrementalBar
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import SpectralCoclustering


def plot_matrix_decomp_ACP(matrix):
    """
    Compute the eigenvectors of a matrix, print the 10 first explained variance ratio, and plot the matrix projected on the 2 first eigenvectors.

    Entries :
    ---------
    matrix : squared array-like.
        Matrix to be projected.

    acp : acp model.
        model fitted with ``matrix``.

    Returns :
    ---------
    coord : list
        list of all the eigenvectors.
    """

    acp = PCA(svd_solver='full')

    coord = acp.fit_transform(matrix)
    print(acp.explained_variance_ratio_[:10])
    plt.plot(coord[:, 0], coord[:, 1], 'bo')
    plt.show()
    return coord


def Coclustering_representation(X, subjects, coclust_shape, coclust_method="log"):
    """
    Prints the specified graphical coclustering.

    Entries :
    ---------

    X : array-like of shape (``n_samples``, ``n_features``)
        Matrix of datas NOT REARRANGED.

    subjects : array-like, shape=(``n_samples``, ``1``)
        List of the subjetcs that did the action given to a line of X.

    coclust_shape : pair (``nb of clusters in line``, ``nb of clusters in column``)
        shape of coclustering.

    coclust_method : string, among {\"log\",\"bistochastic\",\"scale\",\"all\"}, default is \"log\".
        method to be used to form the coclustering model.

    Returns :
    ---------
    None, only prints the matrix.
    """

    if coclust_method not in {"log", "bistochastic", "scale", "all"}:
        print(
            "Choice of coclust_method not supported. Please choose among {\"log\",\"bistochastic\",\"scale\",\"all\"}")
        return None

    if coclust_method != "all":

        X_grouped = ajouter_actions(X, subjects, coclust_shape[1])
        repres_model = SpectralBiclustering(
            n_clusters=coclust_shape, method=coclust_method)
        repres_model.fit(X_grouped)

        fit_data = X_grouped[np.argsort(repres_model.row_labels_)]
        fit_data = fit_data[:, np.argsort(repres_model.column_labels_)]
        plt.matshow(fit_data, cmap=plt.get_cmap(
            name='Blues_r'), aspect='auto')
        plt.title("After BICLUSTERING; rearranged to show BICLUSTERS, " +
                  coclust_method+f" {coclust_shape}")

        plt.show()

    else:
        X_grouped = ajouter_actions(X, subjects, coclust_shape[1])
        repres_model = SpectralBiclustering(
            n_clusters=coclust_shape, method='log')
        repres_model.fit(X_grouped)

        fit_data = X_grouped[np.argsort(repres_model.row_labels_)]
        fit_data = fit_data[:, np.argsort(repres_model.column_labels_)]
        plt.matshow(fit_data, cmap=plt.get_cmap(
            name='Blues_r'), aspect='auto')
        plt.title(
            f"After BICLUSTERING; rearranged to show BICLUSTERS, log {coclust_shape}")

        X_grouped = ajouter_actions(X, subjects, coclust_shape[1])
        repres_model = SpectralBiclustering(n_clusters=coclust_shape)
        repres_model.fit(X_grouped)

        fit_data = X_grouped[np.argsort(repres_model.row_labels_)]
        fit_data = fit_data[:, np.argsort(repres_model.column_labels_)]
        plt.matshow(fit_data, cmap=plt.get_cmap(
            name='Blues_r'), aspect='auto')
        plt.title(
            f"After BICLUSTERING; rearranged to show BICLUSTERS, bistochastic {coclust_shape}")

        X_grouped = ajouter_actions(X, subjects, coclust_shape[1])
        repres_model = SpectralBiclustering(
            n_clusters=coclust_shape, method='scale')
        repres_model.fit(X_grouped)

        fit_data = X_grouped[np.argsort(repres_model.row_labels_)]
        fit_data = fit_data[:, np.argsort(repres_model.column_labels_)]
        plt.matshow(fit_data, cmap=plt.get_cmap(
            name='Blues_r'), aspect='auto')
        plt.title(
            f"After BICLUSTERING; rearranged to show BICLUSTERS, scale {coclust_shape}")

        plt.show()

    return repres_model


def biclustering_impact_viewer(array, nb_row_clusters, nb_col_clusters, matrix_name):
    """Plots the matrix using matplotlib in its original version and in its biclustered version in the same window
    The clusters are highlighted by edges drawn after biclustering

    Params:
    -------
    array           :       2D numpy array to bicluster using SpectralBiclustering
    nb_row_clusters :       number of rows of clusters
    nb_col_clusters :       number of columns of clusters
    matrix_name     :       the name of the matrix to be plotted as a title

    Returns:
    --------
    bicluster       :       SpectralBiclustering object after fitting the matrix of data
    """
    # Initializing the clustering process
    cluster = SpectralBiclustering(
        n_clusters=(nb_row_clusters, nb_col_clusters))
    cluster.fit(array)
    # Reorganizing the rows and columns so that the clusters appear
    fit_array = array[np.argsort(cluster.row_labels_)]
    fit_array = fit_array[:, np.argsort(cluster.column_labels_)]

    plt.suptitle(matrix_name)
    # Plots the original matrix to bicluster
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.matshow(array, cmap=plt.get_cmap(
        name='plasma'), aspect='auto', fignum=0)
    plt.colorbar()
    # Plots the matrix after biclustering
    plt.subplot(1, 3, 2)
    plt.title("Biclustered")
    plt.matshow(fit_array, cmap=plt.get_cmap(
        name='plasma'), aspect='auto', fignum=0)
    plt.colorbar()

    x, y = -0.5, -0.5   # Drawing lines to separate clusters
    for i in range(nb_row_clusters):
        for j in range(nb_col_clusters):
            n = j + i*nb_col_clusters
            ix, iy = cluster.get_shape(n)
            plt.plot([y, y+iy], [x+ix, x+ix], "g")
            plt.plot([y+iy, y+iy], [x, x+ix], "g")
            y += iy
        y = -0.5
        x += ix

    # Plots the same matrix after biclustering by replacing the values in each cluster with the mean value in the cluster (visually better)
    mean_array = array[:, :]
    print(mean_array.shape)
    for i in range(nb_col_clusters*nb_row_clusters):
        # cluster i
        row_indices, column_indices = cluster.get_indices(i)
        sub = cluster.get_submatrix(i, mean_array)
        m = np.mean(sub)
        mean_array[np.ix_(row_indices, column_indices)] = m

    fit_mean_array = mean_array[np.argsort(cluster.row_labels_)]
    fit_mean_array = fit_mean_array[:, np.argsort(cluster.column_labels_)]

    plt.subplot(1, 3, 3)
    plt.title("Biclustered mean matrix")
    plt.matshow(fit_mean_array, cmap=plt.get_cmap(
        name='plasma'), aspect='auto', fignum=0)
    plt.colorbar()

    x, y = -0.5, -0.5
    for i in range(nb_row_clusters):
        for j in range(nb_col_clusters):
            n = j + i*nb_col_clusters
            ix, iy = cluster.get_shape(n)
            plt.plot([y, y+iy], [x+ix, x+ix], "g")
            plt.plot([y+iy, y+iy], [x, x+ix], "g")
            y += iy
        y = -0.5
        x += ix

    # End
    plt.show()

    return cluster


def biclustering_impact_viewer_bis(array, nb_row_clusters, nb_col_clusters, matrix_name):
    """Plots the matrix using matplotlib in its original version and in its biclustered version in the same window
    The clusters are highlighted by edges drawn after biclustering

    Params:
    -------
    array           :       2D numpy array to bicluster using SpectralBiclustering
    nb_row_clusters :       number of rows of clusters
    nb_col_clusters :       number of columns of clusters
    matrix_name     :       the name of the matrix to be plotted as a title

    Returns:
    --------
    bicluster       :       SpectralBiclustering object after fitting the matrix of data
    fit_array       :       array of bicluster
    """
    # Initializing the clustering process
    cluster = SpectralBiclustering(
        n_clusters=(nb_row_clusters, nb_col_clusters))
    cluster.fit(array)
    # Reorganizing the rows and columns so that the clusters appear
    fit_array = array[np.argsort(cluster.row_labels_)]
    fit_array = fit_array[:, np.argsort(cluster.column_labels_)]

    plt.suptitle(matrix_name)
    # Plots the original matrix to bicluster
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.matshow(array, cmap=plt.get_cmap(
        name='plasma'), aspect='auto', fignum=0)
    plt.colorbar()
    # Plots the matrix after biclustering
    plt.subplot(1, 3, 2)
    plt.title("Biclustered")
    plt.matshow(fit_array, cmap=plt.get_cmap(
        name='plasma'), aspect='auto', fignum=0)
    plt.colorbar()

    x, y = -0.5, -0.5   # Drawing lines to separate clusters
    for i in range(nb_row_clusters):
        for j in range(nb_col_clusters):
            n = j + i*nb_col_clusters
            ix, iy = cluster.get_shape(n)
            plt.plot([y, y+iy], [x+ix, x+ix], "g")
            plt.plot([y+iy, y+iy], [x, x+ix], "g")
            y += iy
        y = -0.5
        x += ix

    # Plots the same matrix after biclustering by replacing the values in each cluster with the mean value in the cluster (visually better)
    mean_array = array[:, :]
    print(mean_array.shape)
    for i in range(nb_col_clusters*nb_row_clusters):
        # cluster i
        row_indices, column_indices = cluster.get_indices(i)
        sub = cluster.get_submatrix(i, mean_array)
        m = np.mean(sub)
        mean_array[np.ix_(row_indices, column_indices)] = m

    fit_mean_array = mean_array[np.argsort(cluster.row_labels_)]
    fit_mean_array = fit_mean_array[:, np.argsort(cluster.column_labels_)]

    plt.subplot(1, 3, 3)
    plt.title("Biclustered mean matrix")
    plt.matshow(fit_mean_array, cmap=plt.get_cmap(
        name='plasma'), aspect='auto', fignum=0)
    plt.colorbar()

    x, y = -0.5, -0.5
    for i in range(nb_row_clusters):
        for j in range(nb_col_clusters):
            n = j + i*nb_col_clusters
            ix, iy = cluster.get_shape(n)
            plt.plot([y, y+iy], [x+ix, x+ix], "g")
            plt.plot([y+iy, y+iy], [x, x+ix], "g")
            y += iy
        y = -0.5
        x += ix

    # End
    plt.show()

    # fit_neuron = find_neuron(array, cluster)

    return cluster, fit_array


def find_neuron(array, cluster):
    x = np.arange(0, cluster.column_labels_.shape[0])
    y = np.arange(0, cluster.row_labels_.shape[0])
    neuron_label, _ = np.meshgrid(x, y)

    fit_neuron = neuron_label[np.argsort(cluster.row_labels_)]
    fit_neuron = fit_neuron[:, np.argsort(cluster.column_labels_)]
    return fit_neuron
