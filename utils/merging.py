###########################################################################################################
######################        Merging : Aymeric        ####################################################
###########################################################################################################
import numpy as np
from math import inf
from random import shuffle, randint
from utils.io import *


def extract_patient(X, Z):
    """
    Entry :  X the matrix of measures (x_train+x_test)
             Z the matrix of the subjects (subject_train+subject_test)
    Output : Temp a list of of list where one list contains all the X measures coresponding to one subject in particular
    """

    Temp = [[] for i in range(30)]
    for i in range(len(X)):

        Temp[int(Z[i][0]) - 1].append(X[i])
    return Temp


def merge(Patient, nombre_activite):
    """
    Entry : Patient : a list of all the activities X measures made by one subject (i.e one element of extract_patient (X,Z))
            nombre_activite : the number of activities to concatenate
    Out : Temp : a numpy matrix of shape ((len(Patient),nombre_activite,len(Patient[0]))) a numpy matrix containing\
         all the X measures of a patient rearanged to have nombre_activite
    """
    n = len(Patient)//nombre_activite
    # subdicision of Patient in nombre_activite arrrays
    M = [np.array(Patient[i*n:(i+1)*n]) for i in range(nombre_activite)]

    t = np.shape(M[0])
    t = (t[0], nombre_activite, t[1])
    Temp = np.zeros(t)

    for i in range(n):
        Temp[i] = [x[i] for x in M]

    return Temp


def ajouter_actions(X, Z, nbre_action, shuffle=False):
    """
    Transform a data matrix so that she now contain nbre_action activities by row.

    Entries :
    ---------
    X : array-like, shape=(``n_indiv``, ``561``)
        The matrix of measures with one activity per row.
    Z : array-like, shape=(``n_indiv``, ``1``)
        The matrix of the subjects.
    nbre_action : int.
        The number of activities to concatenate.
    Returns :
    --------
    C : array-like, shape=(``n_indiv``, ``nbre_action*561``).
        Numpy matrix containing nbre_action activities by row.
    """

    Extracted = extract_patient(X, Z)
    if shuffle:
        np.random.shuffle(Extracted)

    k = 0
    lmax = 0
    T = []
    for i in range(30):
        # looking for the subject with the most activities
        currentL = len(Extracted[i])
        if currentL > lmax:
            k = i
            lmax = currentL
    temp = 0
    while lmax % nbre_action != 0:
        # adding measures (by copying and pasting the first ones) so that lmax = 0 mod nombre_action
        Extracted[k].append(Extracted[k][temp])
        temp += 1
        lmax += 1
    for i in range(30):
        # same process then previously but for all the subjects
        currentL = len(Extracted[i])
        for j in range(lmax - currentL):
            Extracted[i].append(Extracted[i][j])
        # once it is done we put in T[i] a numpy matrix of shape (n1, nbre_action,n2)
        T.append(merge(Extracted[i], nbre_action))
    # C is the final matrix of shape (len(X)//nbre_action , len(X[0])* nbre_action)
    C = T[0]
    for i in range(1, 30):
        C = np.concatenate((C, T[i]))

    t = np.shape(C)
    # here we reshape C corresponding to the precedent comment
    C = np.reshape(C, (t[0], t[2]*nbre_action))
    return C


def group_by_subject(X, subjects, nb=inf):
    """
    Group a X matrix by concatenating nb lines from a same subject

    Parameters :
    ------------
    X : matrix of data

    subjects : list of each subject for one line in X

    nb : number of lines to concatenate, default is all (inf)

    Returns :
    ---------
    A new matrix with less lines and more columns than X
    """
    # Creating a dict where a key is a user and values are lists of actions related to this user
    subj_datas = {}
    for i in range(len(X)):
        subj = subjects[i, 0]
        if subj in subj_datas:
            subj_datas[subj].append(X[i])
        else:
            subj_datas[subj] = [X[i]]

    # Shuffling all the datas related to each user in order to avoid overfitting
    for subj in subj_datas:
        shuffle(subj_datas[subj])

    # if no number of lines is given, we state that this is the max number of concatenations possible
    if nb == inf:
        nb = max([len(x) for x in subj_datas.values()])

    # Completing the number of datas in case one user doesn't have a number of lines that is divisible by nb
    for subj in subjects[:, 0]:
        length = len(subj_datas[subj])
        while len(subj_datas[subj]) % nb != 0:
            subj_datas[subj].append(subj_datas[subj][randint(0, length-1)])

    # Creating the new matrix hosting the rearranged datas
    keys = list(subj_datas.keys())
    nb_lines = int(sum([len(subj_datas[subj])/nb for subj in keys]))
    grouped_X = np.zeros((nb_lines, len(X[0])*nb))

    # rearranging datas :
    # s is the index of the subject
    # j is the line in the grouped_X matrix
    s, j = 0, 0
    while j < np.shape(grouped_X)[0]:
        for i in range(nb):
            grouped_X[j, : (i+1)*561] = np.concatenate(
                (grouped_X[j, :i*561], subj_datas[keys[s]].pop(0)))

        if subj_datas[keys[s]] == []:
            s = s+1
        j += 1

    return grouped_X


###########################################################################################################
######################        Merging : Ian Evan        ###################################################
###########################################################################################################


def count_dico(subjects):
    """Returns a dictionary storing the number of rows assignd to each subject number
    Returns:
    --------
    dico[subject_nb] = nb_rows"""
    count = {}

    for x in subjects:
        val = int(x[0])
        if not(val in count):
            count[val] = 0
        count[val] += 1

    return count


def generate_reorder_dico(subjects, multiplicity):
    """From a subjects-numpy 2D-array of shape (n,1) returns a dictionnary
    describing the reordering function of the lines of the array to create a new array concatening <multiplicity> lines in a row"""
    dico = {}

    # counts the number of lines for each subject
    count = count_dico(subjects)

    # new numbers of lines for the new array to create
    n_lines = {i: count[i]//multiplicity for i in count}

    cumul = [0]
    for i in n_lines:
        cumul.append(cumul[-1]+n_lines[i])

    cumul_dico = {list(count.keys())[i]: cumul[i] for i in range(
        len(list(n_lines.keys())))}    # cumulated sums of n_lines

    # register of writing cursors for each subject
    prompt = {i: 0 for i in count}
    nb_rows = int(subjects.shape[0])
    # for each row matches the corresponding subject
    subj_dico = {row: int(subjects[row, 0]) for row in range(nb_rows)}

    l = list(range(nb_rows))
    # shuffles the lines to have a randomized traveling path
    shuffle(l)

    for i_row in l:
        subject = subj_dico[i_row]
        if prompt[subject]//multiplicity < n_lines[subject]:
            # concatenates <multiplicity> lines into one and ignores the lines that cannot be fit in under the good format without adding redundancy
            dico[i_row] = cumul_dico[subject] + prompt[subject]//multiplicity
            prompt[subject] += 1

    return dico


def concatenate_data_lines(data, subjects, nb_lines):
    """Concatenates nb_lines of data into one single line of a new matrix with the constraint
    that all lines that are being concatenated correspond to the same subject specified in the column matrix subjects

    Returns:
    --------
    new_data : same type as data with less than len(data)//nb_lines lines"""
    reorder_function = generate_reorder_dico(subjects, nb_lines)
    m = max(reorder_function.values())+1
    new_data = [None]*m
    for i in range(data.shape[0]):
        if i in reorder_function:
            if new_data[reorder_function[i]] == None:
                new_data[reorder_function[i]] = []
            new_data[reorder_function[i]].extend(list(data[i]))
    return np.array(new_data)


if __name__ == "__main__":
    activitiesFile = "original_data/train/y_train.txt"
    dataFile = "original_data/train/X_train.txt"
    featuresFile = "original_data/features.txt"

    ans = DF1("original_data/train/X_train.txt",
              "original_data/train/y_train.txt", "original_data/features.txt", False)

    data = readFile(dataFile)
    print(data)

