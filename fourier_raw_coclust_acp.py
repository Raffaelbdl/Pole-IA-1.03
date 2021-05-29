
from utils.io import MergeAllInertialData, readFile
from utils.tools import plot_matrix_decomp_ACP, biclustering_impact_viewer
from utils.scoring import get_best_score, Alt_CH_biclust, Alt_silhouette_biclust, cluster_activity_score, evaluate_clustering
import numpy as np
from scipy.fftpack import fft
import seaborn as sn
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn.decomposition import PCA
import json


def freq_stat_analysis(origin_freq_array, clustering_fit):
    """
    Params:
    -------
    origin_freq_array   :   numpy array with original frequencies on a row
    clustering_fit      :   biclustering object after using fit on data

    Returns:
    --------
    stats       :   numpy array of shape (row_clusters, col_clusters) filled with dictionaries
                    representing the statistical analysis of frequencies in each column cluster as followed :
                    scores[j] = {mean : 0.5, std : 0.2}
    """
    row_clusters, col_clusters = np.max(clustering_fit.row_labels_) + \
        1, np.max(clustering_fit.column_labels_)+1
    stats = np.empty(col_clusters, dtype=dict)
    for i in range(row_clusters):
        for j in range(col_clusters):
            n = j+i*col_clusters    # index of the cluster
            _, col_indices = clustering_fit.get_indices(n)
            freqs = np.array([origin_freq_array[f] for f in col_indices])
            stats[j] = {"mean": np.mean(freqs), "std": np.std(freqs)}

    return stats


if __name__ == '__main__':
    inertial_data = MergeAllInertialData()  # Retrieve all the time series

    n_row, n_col, _ = inertial_data.shape   # shape : (n_row, n_col, 6)

    Dt = 2.56  # s

    f_ech = n_col/Dt

    # Creation of a  PCA reduction solver to reduce the 6 components to the first most explicative eigenvector
    acp = PCA(n_components=1, svd_solver='full')

    new_arrays = []
    acp.fit(inertial_data[:, 0, :])                 # Fit the ACP on the data corresponding to t=0
    print("ACP components :", acp.n_components)

    for j in range(n_col):
        # print(inertial_data[:, j, :].shape)

        # Uncomment next line to make adaptative eigenvectors
        # acp.fit(inertial_data[:, j, :])
        new_array = acp.transform(inertial_data[:, j, :])
        # print(new_array.shape)
        new_arrays.append(new_array)

    new_inertial = np.concatenate(new_arrays, axis=1)

    print("New shape (reduced) : ", new_inertial.shape)

    ts = np.linspace(0, 2.56, new_inertial.shape[1])
    xs = new_inertial[0, :]

    xs0 = inertial_data[0, :, 0]

    plt.plot(ts, xs0, 'r')
    plt.title("Ligne temporelle avant ACP")
    plt.xlabel("temps (s)")
    plt.ylabel("Body_acc_x")
    plt.show()

    plt.plot(ts, xs, 'r')
    plt.title("Ligne temporelle après ACP")
    plt.xlabel("temps (s)")
    plt.ylabel("ACP projection (-)")
    plt.show()

    C = new_inertial

    F = 2*np.abs(fft(new_inertial, axis=1)[:, :n_col//2])/n_col  # Performing fft on the time axis

    print("Shape of the fft matrix : ", F.shape)

    fs = np.linspace(0, f_ech/2, F.shape[1])

    # computing a spectral  biclustering

    row_clusters, col_clusters = 6, 4

    plt.matshow(C, cmap=plt.get_cmap(name='plasma'), aspect='auto')
    plt.title("Original time matrix with PCA reduction")
    plt.show()

    cluster = biclustering_impact_viewer(F, row_clusters, col_clusters, "Freq matrix with PCA reduction")

    # print(n_row, n_col)
    # print(cluster.get_shape(0))
    # print(cluster.get_indices(0))

    # X_test = readFile("../UCI HAR Dataset/test/X_test.txt")

    y_test = readFile("working_data/test/y_test.txt")

    subject_test = readFile("working_data/test/subject_test.txt")

    # X_train = readFile("../UCI HAR Dataset/train/X_train.txt")

    y_train = readFile("working_data/train/y_train.txt")

    subject_train = readFile("working_data/train/subject_train.txt")

    subjects = np.concatenate((subject_train, subject_test))
    activities = np.concatenate((y_test, y_train))

    # X = np.concatenate((X_train, X_test))

    ##################### SEARCH OF THE BEST SCORE ######################

    get_best_score(F, subjects, 30, Alt_silhouette_biclust, "max")

    get_best_score(F, subjects, 30, Alt_CH_biclust, "max")

    plt.show()

    ##################### CLUSTERS SCORES #####################

    cluster_activities = evaluate_clustering(activities, cluster)
    cluster_act_score = cluster_activity_score(activities, cluster)

    for i in range(row_clusters):
        for j in range(col_clusters):
            print("Activities in cluster ({},{}) : ".format(i, j), cluster_activities[i, j])
    print("Cluster mean activity score : ", cluster_act_score)

    stat_analysis = freq_stat_analysis(fs, cluster)

    for j in range(col_clusters):
        print("Column {} : mean = {}, std = {}".format(j, stat_analysis[j]["mean"], stat_analysis[j]["std"]))
