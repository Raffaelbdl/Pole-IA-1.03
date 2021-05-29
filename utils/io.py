import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def readFile(filepath):
    '''
    Inputs:
        filepath : filepath
    Outputs:
        m : numpy matrix representing the data
    '''
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.to_numpy()


def writeFile(filepath, matrix: np.ndarray):
    '''
    Inputs:
        filepath : filepath
        matrix : numpy array
    '''
    dataframe = pd.DataFrame(matrix)
    dataframe.to_csv(filepath, header=None, index=None, sep=' ', mode='w')


def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = readFile(prefix + name)
        loaded.append(data)
    loaded = np.dstack(loaded)
    return loaded


def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial_Signals/'
    filenames = list()
    filenames += ['total_acc_x_'+group+'.txt',
                  'total_acc_y_'+group+'.txt',
                  'total_acc_z_'+group+'.txt']
    filenames += ['body_acc_x_'+group+'.txt',
                  'body_acc_y_'+group+'.txt',
                  'body_acc_z_'+group+'.txt']
    filenames += ['body_gyro_x_'+group+'.txt',
                  'body_gyro_y_'+group+'.txt',
                  'body_gyro_z_'+group+'.txt']
    X = load_group(filenames, filepath)
    y = readFile(prefix + group + '/y_' + group + '.txt')
    return X, y


def load_dataset(prefix='./original_data/'):
    trainX, trainy = load_dataset_group('train', prefix)

    testX, testy = load_dataset_group('test', prefix)

    trainy = trainy - 1
    testy = testy - 1

    trainy = tf.keras.utils.to_categorical(trainy)
    testy = tf.keras.utils.to_categorical(testy)

    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


# Fonctions pour visualiser les activités

def one_activity_xyz(X, Y, Z, first_line, last_line):
    '''
    Inputs:
        X, Y, Z are the lists of time series, each line associated to an intervalle of an activity
        first_line is the first line concatenated
        last_line is the last line concatenated
    Outputs:
        _x, _y, _z are flattened vectors that contain a single time serie (hopefully from a single activity)
    '''

    _x = np.array(X[first_line: last_line]).ravel()
    _y = np.array(Y[first_line: last_line]).ravel()
    _z = np.array(Z[first_line: last_line]).ravel()

    return _x, _y, _z


def all_activity_x(X, labels):
    '''
    Inputs:
        X, is the lists of time series, each line associated to an intervalle of an activity
        labels is the list that contain the type of activity executed for each line of X
    Outputs: 
        _X is the list where each line contain the full activity flattened
        _labels is the list that contain the type of activity executed for each line of _X
    '''

    cur = 0
    _cur = cur

    _X = []
    _labels = []

    while _cur < len(labels):
        if labels[_cur] != labels[cur]:
            _x = np.concatenate(X[cur: _cur])
            _X.append(_x)
            _labels.append(labels[cur])
            cur = _cur
        else:
            _cur += 1

    return _X, _labels


def all_activity_xyz(X, Y, Z, labels):
    '''
    Inputs:
        X, Y, Z are the lists of time series, each line associated to an intervalle of an activity
        labels is the list that contain the type of activity executed for each line of X, Y and Z
    Outputs: 
        _X, _Y, _Z are the lists where each line contain the full activity flattened
        _labels is the list that contain the type of activity executed for each line of _X, _Y and _Z
    '''

    cur = 0
    _cur = cur

    _X = []
    _Y = []
    _Z = []
    _labels = []

    while _cur < len(labels):
        if labels[_cur] != labels[cur]:
            _x, _y, _z = one_activity_xyz(X, Y, Z, cur, _cur)
            _X.append(_x)
            _Y.append(_y)
            _Z.append(_z)
            _labels.append(labels[cur])
            cur = _cur
        else:
            _cur += 1

    return _X, _Y, _Z, _labels


def array2DF(array, columnLabels):
    """Transforms the numpy matrix to a panda DataFrame using columnLabels (list)

    Params:
    -------
    array           :   numpy array (matrix)
    columnLabels    :   labels of the columns (list)

    Returns:
    --------
    df              :   panda DataFrame"""

    df = pd.DataFrame(array, columns=columnLabels)
    return df


def DF1(dataFile, activitiesFile, featuresFile, indexingColumns=False):
    """Returns a complete DataFrame of activities and data vectors with column names

    Params:
    -------
    dataFile        :   file path to data file (X_train)
    activitiesFile  :   file path to activities file (y_train)
    featuresFile    :   file path to features file (features)
    indexingColumns :   boolean indicates whether to label columns using integers (True) or names (False)

    Return:
    -------
    df              :   panda DataFrame"""

    acts = readFile(activitiesFile, int)
    xtrain = readFile(dataFile, float)
    features = readFile(featuresFile, str)
    df = array2DF(acts, ['Activity'])
    df2 = array2DF(xtrain, features[:, 0 if indexingColumns else 1])
    train = pd.concat([df, df2], axis=1)
    return train


def MergeBodyAcc(normalize=False):
    """Takes x,y,z body acc matrix  and returns a matrix of vectors of [x,y,z] body acc"""
    X_test = readFile(
        "original_data/test/Inertial_Signals/body_acc_x_test.txt")
    Y_test = readFile(
        "original_data/test/Inertial_Signals/body_acc_y_test.txt")
    Z_test = readFile(
        "original_data/test/Inertial_Signals/body_acc_z_test.txt")
    X_train = readFile(
        "original_data/train/Inertial_Signals/body_acc_x_train.txt")
    Y_train = readFile(
        "original_data/train/Inertial_Signals/body_acc_y_train.txt")
    Z_train = readFile(
        "original_data/train/Inertial_Signals/body_acc_z_train.txt")

    X = np.concatenate((X_test, X_train))
    Y = np.concatenate((Y_test, Y_train))
    Z = np.concatenate((Z_test, Z_train))
    n, m = np.shape(X)
    M = np.zeros((n, m, 3))
    for i in range(n):
        for j in range(m):
            M[i][j][:] = [X[i][j], Y[i][j], Z[i][j]]

    if normalize:
        M = np.linalg.norm(M, axis=2)
    return M


def MergeBodyGyro(normalize=False):
    """Takes x,y,z body acc matrix  and returns a matrix of vectors of [x,y,z] body acc"""
    X_test = readFile(
        "original_data/test/Inertial_Signals/body_gyro_x_test.txt")
    Y_test = readFile(
        "original_data/test/Inertial_Signals/body_gyro_y_test.txt")
    Z_test = readFile(
        "original_data/test/Inertial_Signals/body_gyro_z_test.txt")
    X_train = readFile(
        "original_data/train/Inertial_Signals/body_gyro_x_train.txt")
    Y_train = readFile(
        "original_data/train/Inertial_Signals/body_gyro_y_train.txt")
    Z_train = readFile(
        "original_data/train/Inertial_Signals/body_gyro_z_train.txt")

    X = np.concatenate((X_test, X_train))
    Y = np.concatenate((Y_test, Y_train))
    Z = np.concatenate((Z_test, Z_train))
    n, m = np.shape(X)
    M = np.zeros((n, m, 3))
    for i in range(n):
        for j in range(m):
            M[i][j][:] = [X[i][j], Y[i][j], Z[i][j]]

    if normalize:
        M = np.linalg.norm(M, axis=2)

    return M


def MergeAllInertialData():
    """ Reads all the inertial data files (body_acc and body_gyro along all axes) from test/ and train/ and stacks them along a new axis

    Returns:
    --------
    data        :       3-dim np.array of shape (n,m,6)"""
    file_paths_test = [s for s in os.listdir(
        "working_data/test/Inertial Signals") if s[0] == 'b']
    file_paths_train = [s for s in os.listdir(
        "working_data/train/Inertial Signals") if s[0] == 'b']
    arrays = [np.concatenate((readFile("working_data/test/Inertial Signals/"+file_paths_test[i]),
                              readFile("working_data/train/Inertial Signals/"+file_paths_train[i]))) for i in range(6)]
    data = np.stack(arrays, axis=2)
    return data
