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


def evaluate_clustering(activities_array, clustering_fit):
    """
    Params:
    -------
    activities_array    :   numpy array with original activities on rows
    clustering_fit      :   biclustering object after using fit on data

    Returns:
    --------
    scores      :   numpy array of shape (row_clusters, col_clusters) filled with dictionaries
                    representing the repartition of activities in each cluster as followed :
                    scores[i,j] = {1: 0, 2: 0.2, ..., 6 : 0.8}
    """
    row_clusters, col_clusters = np.max(clustering_fit.row_labels_) + \
        1, np.max(clustering_fit.column_labels_)+1
    scores = np.empty((row_clusters, col_clusters), dtype=dict)
    for i in range(row_clusters):
        for j in range(col_clusters):
            n = j+i*col_clusters    # index of the cluster
            row_indices, _ = clustering_fit.get_indices(n)
            tot, _ = clustering_fit.get_shape(n)
            # increment to calculate the proportions of activities in each cluster
            increment = 1.0/tot
            scores[i, j] = dict()
            for r in row_indices:
                act_ind = int(activities_array[int(r)])
                if not(act_ind in scores[i, j]):
                    scores[i, j][act_ind] = 0.0
                scores[i, j][act_ind] += increment
    return scores


def better_score(row_score_array: np.ndarray, LABELS: dict):
    output = [dict() for _ in range(row_score_array.shape[0])]
    for i in range(row_score_array.shape[0]):
        for j in range(1, len(LABELS.values())+1):
            try:
                if row_score_array[i][0][j] >= 1/6:
                    output[i][LABELS[j]] = int(row_score_array[i][0][j] * 100)
            except:
                pass
    return output


def cluster_activity_score(activities_array, clustering_fit):
    """
    Params:
    -------
    activities_array    :   numpy array with original activities on rows
    clustering_fit      :   biclustering object after using fit on data

    Returns:
    --------
    score (float)       :   a unique score which is the average value of the maximum score in each cluster, 
                            the weights are the sizes of the clusters
    """
    row_clusters, col_clusters = np.max(clustering_fit.row_labels_) + \
        1, np.max(clustering_fit.column_labels_)+1
    scores = evaluate_clustering(activities_array, clustering_fit)
    vals = []
    weights = []
    for i in range(row_clusters):
        for j in range(col_clusters):
            n = j+i*col_clusters    # index of the cluster
            n_rows, n_cols = clustering_fit.get_shape(n)
            weights.append(n_rows*n_cols)
            vals.append(max(scores[i, j].values()))
    return float(np.average(np.array(vals), weights=np.array(weights)))


def silhouette_biclust(Bicluster, data):
    """
    Compute the silhouette score for an entire bicluster (mean of all the silhouette scores for each sample).

    Entries :
    ---------
    Bicluster : bicluster model
        Model already fitted with data.

    data : array-like, shape=(``n_samples``, ``n_features``)
        Matrix of datas NOT REARRANGED related to Bicluster by ``Bicluster.fit(data)``.

    Returns :
    ---------
    silhouette : float.
        silhouette score of the biclustering model.
    """

    (n, c) = np.shape(data)
    L, K = np.max(Bicluster.row_labels_)+1, np.max(Bicluster.column_labels_)+1
    row_labels, column_labels = Bicluster.row_labels_, Bicluster.column_labels_

    t_data = np.transpose(data)

    Silh_mat = np.zeros((K, L))

    for k in range(K):
        for l in range(L):

            k_labels = np.array([int(i == k) for i in column_labels])
            l_labels = np.array([int(j == l) for j in row_labels])

            s1 = silhouette_score(data, l_labels)

            s2 = silhouette_score(t_data, k_labels)

            Silh_mat[k, l] = (n*s1+c*s2)/(n*c)

    return float(np.mean(Silh_mat))


def Alt_silhouette_biclust(Bicluster, data):
    """
    Alternative silhouette score

    Entries :
    ---------

    Bicluster : bicluster model
        Model already fitted with data.

    data : array-like, shape=(``n_samples``, ``n_features``)
        Matrix of datas NOT REARRANGED related to Bicluster by ``Bicluster.fit(data)``.

    Returns :
    ---------
    CH : float.
        Calinski-Harabasz score of the biclustering model.
    """

    (n, c) = np.shape(data)

    row_labels, column_labels = Bicluster.row_labels_, Bicluster.column_labels_
    rc, cc = np.max(Bicluster.row_labels_) + \
        1, np.max(Bicluster.column_labels_)+1

    t_data = np.transpose(data)

    s1 = silhouette_score(data, row_labels)*rc
    s2 = silhouette_score(t_data, column_labels)*cc

    return ((n-rc)*s1+(c-cc)*s2)/(n*c)


###     Calinski-Harabasz score     ###

def CH_biclust1(Bicluster, data):
    """
    Compute the Calinski-Harabasz score for an entire bicluster.

    Entries :
    ---------

    Bicluster : bicluster model
        Model already fitted with data.

    data : array-like, shape=(``n_samples``, ``n_features``)
        Matrix of datas NOT REARRANGED related to Bicluster by ``Bicluster.fit(data)``.

    Returns :
    ---------
    CH : float.
        Calinski-Harabasz score of the biclustering model.
    """

    def norm_mat_euc(A):  # Fonction outil
        return np.sum(A**2)

    rows, columns = np.transpose(
        Bicluster.rows_), np.transpose(Bicluster.columns_)
    (n, d) = np.shape(data)

    x_bar = np.mean(data)

    g_max, m_max = np.max(Bicluster.row_labels_) + \
        1, np.max(Bicluster.column_labels_)+1
    CH = np.zeros((g_max, m_max))

    for l in range(g_max):
        for k in range(m_max):

            z, w = np.zeros((n, g_max)), np.zeros((d, m_max))

            for i in range(n):
                # Matrice des appartenances en lignes
                z[i, l] = int(rows[i, l])

            for j in range(d):
                # Matrice des appartenances en colonnes
                w[j, k] = int(columns[j, k])

            A = np.zeros((g_max, m_max))  # Matrice des poids
            data_lk = np.array([[data[i, j] for i in range(n) if rows[i, l]]
                                for j in range(d) if columns[j, k]])
            A[l, k] = np.mean(data_lk)

            zAwt = z.dot(A.dot(np.transpose(w)))

            CH[l, k] = (norm_mat_euc(zAwt-x_bar*np.ones((n, d))) /
                        (g_max*m_max))/(norm_mat_euc(zAwt-data)/(n*d-g_max*m_max))

    return float(np.mean(CH))


def CH_biclust(Bicluster, data):
    """
    Compute the Calinski-Harabasz score for an entire bicluster.

    Entries :
    ---------

    Bicluster : bicluster model
        Model already fitted with data.

    data : array-like, shape=(``n_samples``, ``n_features``)
        Matrix of datas NOT REARRANGED related to Bicluster by ``Bicluster.fit(data)``.

    Returns :
    ---------
    CH : float.
        Calinski-Harabasz score of the biclustering model.
    """

    (n, c) = np.shape(data)
    L, K = np.max(Bicluster.row_labels_)+1, np.max(Bicluster.column_labels_)+1
    row_labels, column_labels = Bicluster.row_labels_, Bicluster.column_labels_

    t_data = np.transpose(data)

    CH_mat = np.zeros((K, L))

    for k in range(K):
        for l in range(L):

            k_labels = np.array([int(i == k) for i in column_labels])
            l_labels = np.array([int(j == l) for j in row_labels])

            ch1 = calinski_harabasz_score(data, l_labels)

            ch2 = calinski_harabasz_score(t_data, k_labels)

            CH_mat[k, l] = (n*ch1+c*ch2)/(n*c)

    return float(np.mean(CH_mat))


def Alt_CH_biclust(Bicluster, data):
    """
    Compute the Calinski-Harabasz score for an entire bicluster.

    Entries :
    ---------

    Bicluster : bicluster model
        Model already fitted with data.

    data : array-like, shape=(``n_samples``, ``n_features``)
        Matrix of datas NOT REARRANGED related to Bicluster by ``Bicluster.fit(data)``.

    Returns :
    ---------
    CH : float.
        Calinski-Harabasz score of the biclustering model.
    """

    (n, c) = np.shape(data)

    row_labels, column_labels = Bicluster.row_labels_, Bicluster.column_labels_
    rc, cc = np.max(Bicluster.row_labels_) + \
        1, np.max(Bicluster.column_labels_)+1

    t_data = np.transpose(data)

    CH1 = calinski_harabasz_score(data, row_labels)*(rc-1)/(n-rc)
    CH2 = calinski_harabasz_score(t_data, column_labels)*(cc-1)/(c-cc)

    return ((n-rc)*CH1+(c-cc)*CH2)/(n*c)


######      Computing the best scores     ######

def get_best_score(X, subjects, nb_indiv, Score_fct, max_or_min, shuffle=True):
    """
    Compute and prints the best score possible among all the possibilities in terms of Coclustering shapes. The score to be computed is to be specified.

    Entries :
    ---------

    X : array-like of shape (``n_samples``, ``n_features``)
        Matrix of datas NOT REARRANGED.

    subjects : array-like, shape=(``n_samples``, ``1``)
        List of the subjetcs that did the action given to a line of X.

    nb_indiv : int
        Number of individus participating.

    Score_fct : fonction
        Fonction that computes the score studied for a single coclustering.

    max_or_min : string among {\"max\", \"min\"}
        Determines if the score is to be minimized or maximized.

    shuffle : Boolean, True by default.
        If on True, shuffles the data reorganisation to avoid overfitting.

    Returns :
    ---------
    best_score : float.
        Best score computed among all the possible coclustering models.

    best_shape : pair (``nb of clusters in line``, ``nb of clusters in column``)
        shape of coclustering corresponding to the best score.
    """

    if max_or_min not in {"max", "min"}:
        print(
            "Choice of max_or_min not supported. Please choose among {\"max\", \"min\"}")
        return None

    biclust_set = []
    score_set = []
    shapes_set = []

    if max_or_min == "max":
        bar = IncrementalBar('Processing', max=(nb_indiv-1)*9)

        for i in range(2, nb_indiv+1):
            for j in range(2, 10+1):
                bar.next()
                X_data = ajouter_actions(X, subjects, j, shuffle)
                bi_model = SpectralBiclustering(n_clusters=(i, j))
                bi_model.fit(X_data)
                biclust_set.append(bi_model)
                score = Score_fct(bi_model, X_data)

                score_set.append(score)
                shapes_set.append((i, j))
        bar.finish()

        plt.plot(np.array([i for i in range(2, nb_indiv+1)]), np.array(
            [score_set[i*9] for i in range(nb_indiv-1)]))
        plt.legend(str(Score_fct))

        best_score = max(score_set)
        print(best_score, shapes_set[score_set.index(
            best_score)], score_set)
        return best_score, shapes_set[score_set.index(
            best_score)]

    if max_or_min == "min":

        bar = IncrementalBar('Processing', max=(nb_indiv-1)*9)

        for i in range(2, nb_indiv+1):
            for j in range(2, 10+1):
                bar.next()
                X_data = ajouter_actions(X, subjects, j, shuffle)
                bi_model = SpectralBiclustering(n_clusters=(i, j))
                bi_model.fit(X_data)
                biclust_set.append(bi_model)
                score = Score_fct(bi_model, X_data)

                score_set.append(score)
                shapes_set.append((i, j))
        bar.finish()

        plt.plot(np.array([i for i in range(2, nb_indiv+1)]), np.array(
            [score_set[i*9] for i in range(nb_indiv-1)]))
        plt.legend(str(Score_fct))

        best_score = min(score_set)
        print(best_score, shapes_set[score_set.index(
            best_score)], score_set)
        return best_score, shapes_set[score_set.index(
            best_score)]


# DAVIES-BOULDIN SCORE TO BE COMPUTED
def DB_score(Bicluster, data):
    """
    Compute the DAVIES-BOULDIN score

    Entries :
    ---------

    Bicluster : bicluster model
        Model already fitted with data.

    data : array-like, shape=(``n_samples``, ``n_features``)
        Matrix of datas NOT REARRANGED related to Bicluster by ``Bicluster.fit(data)``.

    Returns :
    ---------
    DB : float.
        DAVIES-BOULDIN score of the biclustering model.
    """

    (n, c) = np.shape(data)

    row_labels, column_labels = Bicluster.row_labels_, Bicluster.column_labels_
    rc, cc = np.max(Bicluster.row_labels_) + \
        1, np.max(Bicluster.column_labels_)+1

    t_data = np.transpose(data)

    s1 = davies_bouldin_score(data, row_labels)*rc
    s2 = davies_bouldin_score(t_data, column_labels)*cc

    return ((n-rc)*s1+(c-cc)*s2)/(n*c)
