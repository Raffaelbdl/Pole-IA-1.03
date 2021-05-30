from utils.io import readFile
from utils.merging import ajouter_actions
import utils.scoring as us
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import sklearn.cluster
import fastdtw as fdtw
from scipy.spatial.distance import euclidean
import json
import os
from textwrap import wrap



def load_data(path1,path2):
    """
    Initialization of data.
    """
    # Reading test and train data
    brutX = readFile(path1)
    brutXX = readFile(path2)
    Z = readFile("original_data/train/subject_train.txt")
    ZZ = readFile("original_data/test/subject_test.txt")

    # merging test and train
    brutX = np.concatenate((brutX, brutXX))
    Z = np.concatenate((Z, ZZ))

    return brutX,Z


def perform_dtw(C, nb_actions, savefig=(False, "acc_x")):
    """
    Performs a dynamic time warping with a specified number of actions in each row

    Entry : C the matrix of measures
            nb_actions the number of actions which should be on a line
            savefig is a couple of a boolean and a string. If boolean is true, the figures will be saved,
                and the string is the name of the measure made. If false, the string must be filled, but
                figures won't be saved
    Out : Cprime a numpy matrix, which is the transformation of C through DTW (first line is the reference)
    """

    #Performing the DTW
    n = len(C)
    m = len(C[0])
    Cprime = np.zeros((n, m))

    for j in range(m):
        Cprime[0][j] = C[0][j]

    for i in range(1, n):
        _, path = fdtw.fastdtw(C[0], C[i])

        for (C0element, Cielement) in path:
            Cprime[i][C0element] = C[i][Cielement]

        if i % 100 == 0 :
            print('{} : {} actions : line {} done out of {}'.format(savefig[1],nb_actions,i,n))


    #Potentially saving the figure
    if savefig[0]:
        plt.matshow(Cprime, cmap=plt.get_cmap(name='Blues_r'), aspect='auto')
        plt.savefig("outputs/dtw_results/"+savefig[1]+"_dtw_mat_actions_" + str(nb_actions), dpi=300)
        plt.show()

        plt.matshow(C, cmap=plt.get_cmap(name='Blues_r'), aspect='auto')
        plt.savefig("outputs/dtw_results/"+savefig[1]+"_brut_actions_" + str(nb_actions), dpi=300)
        plt.show()


    return Cprime


def perform_coclust(C,Cprime,nb_actions,nb_rows, scoring_function=(us.Alt_silhouette_biclust,"silhouette_score"), savefig = (True,"acc_x")):
    """
    Given two matrices, performs the coclustering for each of them

    Entry : C the matrix of measures
            Cprime the matrix of measures which went through DTW
            nb_actions the number of actions which should be on a line
            nb_rows the number of expected rows for our biclustering
            scoring_function ((function,string)) A couple. The function that we want to use to evaluate the coclusterings.
                    The string is the name of the function
                    WARNING : This function must take as parameters, in order : a bicluster model, fitted with data, and
                    the raw data, without treatment. We should have used them as following before : Bicluster.fit(data)
            savefig is a couple of a boolean and a string. If boolean is true, the figures will be saved,
                and the string is the name of the measure made. If false, the string has no importance, and
                figures won't be saved

    Out : (Cscore,Cprimescore) the couple of the silhouette score of spectral biclustering for C
            and the silhouette score of the spectral biclustering for Cprime
    """

    # computing a spectral  biclustering with 6 row clusters and 2 column clusters for C and Cprime
    BiclustC = sklearn.cluster.SpectralBiclustering(n_clusters=(nb_rows, nb_actions))
    CoclustC = BiclustC.fit(C)
    BiclustCprime = sklearn.cluster.SpectralBiclustering(n_clusters=(nb_rows, nb_actions))
    CoclustCprime = BiclustCprime.fit(Cprime)

    # fit_C will contain the rearranged matrix of C corresponding to the biclustering
    fit_C = C[np.argsort(CoclustC.row_labels_)]
    fit_C = fit_C[:, np.argsort(CoclustC.column_labels_)]
    fit_Cprime = Cprime[np.argsort(CoclustCprime.row_labels_)]
    fit_Cprime = fit_Cprime[:, np.argsort(CoclustCprime.column_labels_)]

    #Create a graph of the biclustered matrix without DTW. Potentially saves it
    name = str(nb_rows)+"-"+str(nb_actions)
    if savefig[0] :
        plt.matshow(fit_C, cmap=plt.get_cmap(name='Blues_r'), aspect='auto')
        plt.title("BiClustered matrix, without DTW")
        plt.savefig("outputs/dtw_results/"+scoring_function[1]+"/"+savefig[1]+"_without_format_" + name, dpi=300)
        plt.close()
    Cscore = scoring_function[0](BiclustC, C)

    #Create a graph of the biclustered matrix with DTW. Potentially saves it
    if savefig[0] :
        plt.matshow(fit_Cprime, cmap=plt.get_cmap(name='Blues_r'), aspect='auto')
        plt.title("BiClustered matrix, with DTW")
        plt.savefig("outputs/dtw_results/"+scoring_function[1]+"/"+savefig[1]+"_with_format_" + name, dpi=300)
        plt.close()
    Cprimescore = scoring_function[0](BiclustCprime, C)

    return (Cscore,Cprimescore)


def compare(max_actions,max_rows,brutX,Z,savefig,scoring_function=us.Alt_silhouette_biclust):
    """
    Main loop : given a maximum number of actions per line and a maximum number of rows, tests each combination
    by calculating its silhouette score (number of rows and number of columns start at 2)

    Entry : max_actions (Int) the maximum number of actions we want to have on a line
            max_rows (Int) the maximum number of rows we want to have
            brutX (NumPy matrix) the matrix of one of the brut values from the smartphone
            Z (NumPy matrix) the matrix of the users labels
            savefig ((Bool,Str)) is a couple of a boolean and a string. If boolean is true, the figures will be saved,
                and the string is the name of the measure made. If false, the figures won't be saved.
                However, the string should ALWAYS be filled, otherwise it doesn't work
                IMPORTANT : This savefig parameter only concerns the biclustered matrices. The silhouette
                indices will ALWAYS be saved in a json file.
            scoring_function (function) the function that we want to use to evaluate the coclusterings
                    WARNING : This function lmust take as parameters, in order : a bicluster model, fitted with data, and
                    the raw data, without treatment. We should have used them as following before : Bicluster.fit(data)

    Out : resultswithout,resultswith (Dict,Dict) the couple of the dictionnaries in which the silhouette scores are stocked.
            Format of the dictionnary : keys are string looking like : "(4;2)" where 4 is the number of lines
            and 2 the number of columns. An additionnal key exists : "Better coclustering", giving the key of the maximal silhouette score
    """

    resultswithout = {}
    resultswith = {}

    for actions in range(2,max_actions+1):
        #for each number of actions, we create the matrices we need
        C = ajouter_actions(brutX, Z, actions)
        DTWAxisList = []
        DTWScoresList = []
        Cprime = perform_dtw(C, actions, (False, savefig[1]))
        for rows in range(2,max_rows+1):
            #for each row number, we perform both coclustering and we w
            print("{} : Beginning loop : {} rows and {} columns".format(savefig[1],rows, actions))
            Scores = perform_coclust(C,Cprime,actions,rows,scoring_function,savefig)
            dictkey = "("+str(rows)+";"+str(actions)+")"
            resultswithout[dictkey] = Scores[0]
            resultswith[dictkey] = Scores[1]
            DTWAxisList.append(rows)
            DTWScoresList.append(Scores[1])
            print("{} : Ending loop : {} rows and {} columns".format(savefig[1], rows, actions))

        #We create and store the graph of the DTW scores
        plt.plot(DTWAxisList,DTWScoresList)
        plt.title("Silhouette score of {} with {} columns".format(savefig[1],actions))
        plt.savefig("outputs/dtw_results/"+savefig[1]+"withDTW_"+str(actions)+"columns")
        plt.close()

    #We calculate the maximum silhouette score without DTW
    for key,value in resultswithout.items() :
        try :
            if value > maxi1 :
                better_shape_without = key
                maxi1 = value
        except :
            maxi1 = value
            better_shape_without = key
    resultswithout["Better coclustering"] = better_shape_without

    #We calculate the maximum silhouette score with DTW
    for key,value in resultswith.items() :
        try :
            if value > maxi2 :
                better_shape_with = key
                maxi2 = value
        except :
            maxi2 = value
            better_shape_with = key
    resultswith["Better coclustering"] = better_shape_with

    #We store both our dictionnaries into json files.
    with open("outputs/dtw_results/"+savefig[1]+"results_without_DTW.json", "w") as outfile:
        json.dump(resultswithout, outfile)
    with open("outputs/dtw_results/"+savefig[1]+"results_with_DTW.json", "w") as outfile:
        json.dump(resultswith, outfile)

    return better_shape_without,better_shape_with


def main(name_list, max_actions = 6,max_rows = 10, scoring_function=us.Alt_silhouette_biclust, savefigures = False):
    """
    Loop testing compare on all the different types of data.

    Entries :   namlelist (List) the list of the names of the different signals that should be considered. WARNING : the data should be stored in
                    original_data/test/Inertial_Signals/[name]_test.txt and in original_data/train/Inertial_Signals/[name]_test.txt
                max_actions (Int) the maximum number of actions allowed on one single row
                max_rows (Int) the maximum number of rows allowed
                scoring_function (function) the function that we want to use to evaluate the coclusterings
                    WARNING : This function lmust take as parameters, in order : a bicluster model, fitted with data, and
                    the raw data, without treatment. We should have used them as following before : Bicluster.fit(data)
                savefigure (Bool) the boolean : do we want to save all the intermediate coclustering figures ?

    Out : results (Dictionnary) a dictionnary in which we stock, for each signal, the best shape with dtw and without dtw
    """
    result = {}

    #We make sure that the fild dtw_results exists.
    directory = 'outputs/dtw_results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    #We execute a basic loop for each signal
    for element in name_list:
        print("Beginning loop for element {}".format(element))
        path1 = "original_data/train/Inertial_Signals/"+element+"_train.txt"
        path2 = "original_data/test/Inertial_Signals/"+element+"_test.txt"
        brutX,Z = load_data(path1,path2)

        better_shape_without,better_shape_with = compare(max_actions,max_rows,brutX,Z,(savefigures,element),scoring_function)

        result[element] = {}
        result[element]["Better shape without DTW"] = better_shape_without
        result[element]["Better shape with DTW"] = better_shape_with

        print("Ending loop for element {}".format(element))

    #Saving the data in a json file
    with open("outputs/dtw_results/DTW_best_shapes.json","w") as outfile :
        json.dump(result, outfile)

    return result



def alt_compare(max_actions,max_rows,brutX,Z,savefig,scoring_function=(us.Alt_silhouette_biclust,"silhouette_score")):
    """
    Main loop : given a maximum number of actions per line and a maximum number of rows, tests each combination
    by calculating its silhouette score (number of rows and number of columns start at 2)

    Entry : max_actions (Int) the maximum number of actions we want to have on a line
            max_rows (Int) the maximum number of rows we want to have
            brutX (NumPy matrix) the matrix of one of the brut values from the smartphone
            Z (NumPy matrix) the matrix of the users labels
            savefig ((Bool,Str)) is a couple of a boolean and a string. If boolean is true, the figures will be saved,
                and the string is the name of the measure made. If false, the figures won't be saved.
                However, the string should ALWAYS be filled, otherwise it doesn't work
                IMPORTANT : This savefig parameter only concerns the biclustered matrices. The silhouette
                indices will ALWAYS be saved in a json file.
            scoring_function ((function,string)) A couple. The function that we want to use to evaluate the coclusterings.
                    The string is the name of the function
                    WARNING : This function must take as parameters, in order : a bicluster model, fitted with data, and
                    the raw data, without treatment. We should have used them as following before : Bicluster.fit(data)

    Out : resultswithout,resultswith (Dict,Dict) the couple of the dictionnaries in which the silhouette scores are stocked.
            Format of the dictionnary : keys are string looking like : "(4;2)" where 4 is the number of lines
            and 2 the number of columns. An additionnal key exists : "Better coclustering", giving the key of the maximal silhouette score
    """

    resultswithout = {}
    resultswith = {}

    result_matrix_with = np.zeros((max_rows-1,max_actions-1))
    result_matrix_without = np.zeros((max_rows-1,max_actions-1))

    for actions in range(2,max_actions+1):
        #for each number of actions, we create the matrices we need
        C = ajouter_actions(brutX, Z, actions)
        Cprime = perform_dtw(C, actions, (False, savefig[1]))
        for rows in range(2,max_rows+1):
            #for each row number, we perform both coclustering and we w
            print("{} : Beginning loop : {} rows and {} columns".format(savefig[1],rows, actions))
            Scores = perform_coclust(C,Cprime,actions,rows,scoring_function,savefig)
            dictkey = "("+str(rows)+";"+str(actions)+")"
            resultswithout[dictkey] = Scores[0]
            resultswith[dictkey] = Scores[1]
            result_matrix_without[rows-2][actions-2] = Scores[0]
            result_matrix_with[rows-2][actions-2] = Scores[1]
            print("{} : Ending loop : {} rows and {} columns".format(savefig[1], rows, actions))

    #We calculate the maximum silhouette score without DTW
    for key,value in resultswithout.items() :
        try :
            if value > maxi1 :
                better_shape_without = key
                maxi1 = value
        except :
            maxi1 = value
            better_shape_without = key
    resultswithout["Better coclustering"] = better_shape_without

    #We calculate the maximum silhouette score with DTW
    for key,value in resultswith.items() :
        try :
            if value > maxi2 :
                better_shape_with = key
                maxi2 = value
        except :
            maxi2 = value
            better_shape_with = key
    resultswith["Better coclustering"] = better_shape_with

    #We store both our dictionnaries into json files.
    with open("outputs/dtw_results/"+scoring_function[1]+"/"+savefig[1]+"results_without_DTW.json", "w") as outfile:
        json.dump(resultswithout, outfile)
    with open("outputs/dtw_results/"+scoring_function[1]+"/"+savefig[1]+"results_with_DTW.json", "w") as outfile:
        json.dump(resultswith, outfile)

    #We display and store the graph of the scores
    row_labels = range(2,max_rows+1)
    col_labels = range(2,max_actions+1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(result_matrix_with)
    plt.xticks(range(max_actions-1), col_labels)
    plt.yticks(range(max_rows-1), row_labels)
    plt.title("Matrix of scores using {}".format(scoring_function[1]))
    plt.colorbar(cax)
    plt.savefig("outputs/dtw_results/"+scoring_function[1]+"/"+savefig[1]+"_shape_matrix_with", dpi=300)
    plt.close(fig)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    cax2 = ax2.matshow(result_matrix_without)
    plt.xticks(range(max_actions-1), col_labels)
    plt.yticks(range(max_rows-1), row_labels)
    plt.title("Matrix of scores using {}".format(scoring_function[1]),loc='center', wrap=True)
    plt.colorbar(cax2)
    plt.savefig("outputs/dtw_results/"+scoring_function[1]+"/"+savefig[1]+"_shape_matrix_without", dpi=300)
    plt.close(fig2)

    return better_shape_without,better_shape_with


def alt_main(name_list, max_actions = 6,max_rows = 10, scoring_function=(us.Alt_silhouette_biclust,"silhouette_score"), savefigures = False):
    """
    Loop testing compare on all the different types of data.

    Entries :   namlelist (List) the list of the names of the different signals that should be considered. WARNING : the data should be stored in
                    original_data/test/Inertial_Signals/[name]_test.txt and in original_data/train/Inertial_Signals/[name]_test.txt
                max_actions (Int) the maximum number of actions allowed on one single row
                max_rows (Int) the maximum number of rows allowed
                scoring_function ((function,string)) A couple. The function that we want to use to evaluate the coclusterings.
                    The string is the name of the function
                    WARNING : This function must take as parameters, in order : a bicluster model, fitted with data, and
                    the raw data, without treatment. We should have used them as following before : Bicluster.fit(data)
                savefigure (Bool) the boolean : do we want to save all the intermediate coclustering figures ?

    Out : results (Dictionnary) a dictionnary in which we stock, for each signal, the best shape with dtw and without dtw
    """
    result = {}

    #We make sure that the fild dtw_results exists.
    directory = 'outputs/dtw_results/'+scoring_function[1]
    if not os.path.exists(directory):
        os.makedirs(directory)

    #We execute a basic loop for each signal
    for element in name_list:
        print("Beginning loop for element {}".format(element))
        path1 = "original_data/train/Inertial_Signals/"+element+"_train.txt"
        path2 = "original_data/test/Inertial_Signals/"+element+"_test.txt"
        brutX,Z = load_data(path1,path2)

        better_shape_without,better_shape_with = alt_compare(max_actions,max_rows,brutX,Z,(savefigures,element),scoring_function)

        result[element] = {}
        result[element]["Better shape without DTW"] = better_shape_without
        result[element]["Better shape with DTW"] = better_shape_with

        print("Ending loop for element {}".format(element))

    #Saving the data in a json file
    with open("outputs/dtw_results/"+scoring_function[1]+"/DTW_best_shapes.json","w") as outfile :
        json.dump(result, outfile)

    return result


def display_scores(actions,rows, name_list):


    for element in name_list:
        print("Beginning loop for element {}".format(element))
        path1 = "original_data/train/Inertial_Signals/"+element+"_train.txt"
        path2 = "original_data/test/Inertial_Signals/"+element+"_test.txt"
        brutX,Z = load_data(path1,path2)

        C = ajouter_actions(brutX, Z, actions)
        Cprime = perform_dtw(C, actions, (False, element))

        BiclustC = sklearn.cluster.SpectralBiclustering(n_clusters=(rows, actions))
        BiclustC.fit(C)
        BiclustCprime = sklearn.cluster.SpectralBiclustering(n_clusters=(rows, actions))
        BiclustCprime.fit(Cprime)

        y_test = readFile("original_data/test/y_test.txt")
        y_train = readFile("original_data/train/y_train.txt")
        activities = np.concatenate((y_train, y_test))

        print(us.evaluate_clustering(activities, BiclustCprime))
        print(us.cluster_activity_score(activities, BiclustCprime))




name_list = ["body_acc_x","body_acc_y","body_acc_z","body_gyro_x","body_gyro_y","body_gyro_z"]

alt_main(name_list,6,30,(us.DB_score,"DaviesBouldin"),False)