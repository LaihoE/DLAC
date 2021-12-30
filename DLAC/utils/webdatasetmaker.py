import webdataset as wds
import time
from hashlib import sha1
import numpy as np
import multiprocessing
import multiprocessing as mp
import shutil
import os

# Creates a webdataset from numpy arrays


def wdswriter(clean, dirty,x,counter):
    os.mkdir(f"shards/{x}")
    # writes clean and dirty
    with wds.ShardWriter(f'shards/{x}/shard-%07d.tar', maxcount=100000) as sink:
        while True:
            if clean.qsize() > 1000 and dirty.qsize() > 1000:
                clean_data = clean.get()
                with counter.get_lock():
                    inx = counter.value
                    counter.value += 1
                sink.write({
                    "__key__": "samplex%07d" % inx,
                    "y.pyd": 0,
                    "X.pyd": clean_data,
                })
                dirty_data = dirty.get()
                with counter.get_lock():
                    inx = counter.value
                    counter.value += 1
                sink.write({
                    "__key__": "samplex%07d" % inx,
                    "y.pyd": 1,
                    "X.pyd": dirty_data,
                })


def clean_putter(dirtyq,cleanq,clean, dirty, x, counter):
    # Reads "clean" numpy arrays and puts into queue
    while not cleanq.empty():
        print("CleanQ Size:",cleanq.qsize())
        while clean.qsize() > 300000:
            time.sleep(0.1)
        file = cleanq.get()
        a = np.load(file)
        a = np.float32(a)
        for i in range(a.shape[0]):
            clean.put(a[i])


def dirty_putter(dirtyq,cleanq,clean, dirty, x, counter):
    # Reads "dirty" numpy arrays and puts into queue
    while not dirtyq.empty():
        while dirty.qsize() > 1000000:
            time.sleep(0.01)
        print("DirtyQ Size:",dirtyq.qsize())
        file = dirtyq.get()
        a = np.load(file)["arr_0"]
        a = a[:, :, 1:]
        a = np.float32(a)
        for i in range(a.shape[0]):
            dirty.put(a[i])


if __name__ == '__main__':
    counter = multiprocessing.Value("l", 0)
    filenames = []
    clean = mp.Queue()
    dirty = mp.Queue()
    cleanq = mp.Queue()
    dirtyq = mp.Queue()
    for i in range(104):
        dirty_folder = "F:/hd5f/dirty_train/"
        files = os.listdir(dirty_folder)
        for i in files:
            dirtyq.put(f"{dirty_folder}{i}")

    clean_folder = "D:/happy/"
    files = os.listdir(clean_folder)
    for i in files:
        cleanq.put(f"{clean_folder}{i}")
    # Puts clean samples in cleanQ
    processes = [mp.Process(target=clean_putter, args=(dirtyq,cleanq,clean,dirty,x, counter)) for x in range(2)]
    for p in processes:
        p.start()
    # Puts dirty samples in dirtyQ
    processes = [mp.Process(target=dirty_putter, args=(dirtyq,cleanq,clean, dirty, x, counter)) for x in range(2)]
    for p in processes:
        p.start()

    time.sleep(3)
    processes = [mp.Process(target=wdswriter, args=(clean, dirty, x, counter)) for x in range(1,20)]
    for p in processes:
        p.start()
    for p in processes:
        print(p, "joined")
        p.join()
        p.terminate()


    # Joins shards together
    for i in range(1, 48):
        files = os.listdir(f"F:/bigdata/shards/{i}")
        for f in files:
            my_file_name = f"F:/bigdata/shards/{i}/{f}".encode()
            hash_filename = sha1(my_file_name).hexdigest()
            os.rename(f"F:/bigdata/shards/{i}/{f}", f"F:/bigdata/shards/{i}/{hash_filename}.tar")
            shutil.move(f"F:/bigdata/shards/{i}/{hash_filename}.tar", f"F:/bigdata/shards/a/{hash_filename}.tar")
    folder = "F:/bigdata/shards/a/"
    files = os.listdir(folder)
    for cnt, i in enumerate(files):
        cnt = format(cnt, "07")
        os.rename(f"{folder}{i}", f"{folder}shard{cnt}.tar")