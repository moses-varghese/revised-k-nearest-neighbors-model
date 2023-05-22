"""Compute the predicted value for a given query based on the data provided by knn_regression."""

import numpy as np
import pandas as pd

def knn_regression(n_neighbors, data, query):
    """ The algorithm returns the predicted value for query, a single numeric value,
        or raises an appropriate exception (such as ValueError) when inappropriate inputs
        are passed. The functions takes 3 parameters:
        (1)  Parameter k, or n_neighbors
        (2)  data: 2-dimensional numpy array of shape (m, n+1). m denotes the number of samples
             and n is the number of variables in each sample. +1 is for the labels in each sample.
        (3)  query: 1 dimensional numpy array of shape (1,n).
    """
    data_rowsize = len(data)
    data_columnsize = len(data[0])-1
    dis = []
    label = []
    dista = []

    if n_neighbors == 0:
        raise ValueError("the number of nearest neighbors cannot be zero")

    if n_neighbors > data_rowsize:
        raise ValueError('''the number of nearest neighbors should be less than or equal to the
                            sample size of the data''')

    if n_neighbors < 0:
        raise ValueError("the number of nearest neighbors cannot be negative")

    if query == []:
        raise TypeError("knn_regression() missing 1 required positional argument: 'query'")

    if n_neighbors == []:
        raise TypeError("knn_regression() missing 1 required positional argument: 'n_neighbors'")

    for j in range(data_rowsize):
        single_dis = (((query[0] - data[j][0])**2 + (query[1] - data[j][1])**2)**(0.5))
        dis.append(single_dis)
        label.append(data[j][data_columnsize])
        array_disval = {'distance':dis,'value':label}
        df2 = pd.DataFrame(array_disval)
        df1 = df2.sort_values(by='distance')


    for each_n_neigbhors in range(n_neighbors):
        req_datapoint = df1.iloc[each_n_neigbhors]['value']
        dista.append(req_datapoint)
    return np.mean(dista)
