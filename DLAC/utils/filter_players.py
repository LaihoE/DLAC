import filter
import os
from multiprocessing import Process, Value, Array, Lock,Process, Manager, Queue
import time
import numpy as np
import pickle

# Filters wanted amount of players and ignores cheaters

def f(lock, inx, d,q):
    folder = 'F:/csgo/npz_clean1/'
    files = os.listdir(folder)
    while True:
        inx = q.get()
        sub_files = files[inx * 250:inx * 250 + 250]
        a = np.concatenate([np.load(f"{folder}{file}")['arr_0'] for file in sub_files])
        with lock:
            a = filter.speed(a)
        a = np.float32(a[:, :, 1:])
        np.save(f'D:/happy/{inx}', a)


if __name__ == '__main__':
    d = {}
    q = Queue()
    a = np.load("cheatersids.npy")
    for i in a:
        d[i] = 5000

    for i in range(5000):
        q.put(i)
    with open('xd.pickle', 'wb') as handle:
        pickle.dump(d, handle)
    with Manager() as manager:
        d = manager.dict()
        lock = Lock()
        ps = []
        for i in range(22):
            p = Process(target=f, args=(lock,i,d,q))
            ps.append(p)
        for p in ps:
            p.start()