import pickle
import numpy as np


def speed(a):
    cdef int i
    cdef double player
    cdef int totalrange
    with open('xd.pickle', 'rb') as handle:
        d = pickle.load(handle)

    print(len(d))
    cdef ashape = a.shape[0]-1
    total = {}
    l = []

    for i in range(ashape):
        player = int(a[i][0][0])
        if player != 0:
            if player not in total:
                total[player] = 1
                l.append(i)
            else:
                if total[player] <= 1000:
                    total[player] += 1
                    l.append(i)
    with open('xd.pickle', 'wb') as handle:
        pickle.dump(d, handle)
    return a[l]
