import numpy as numpy
import pandas as pd
import pickle
def make_train_data():
    f = open("C:\\Users\\yuuya\\Documents\\git\\Blog\\src\\BNN\\transitions.binaryfile", "rb")
    data = pickle.load(f)

    new_list = []
    for l in data:
        new_list.extend(l)
    l = []
    for t in new_list:
        l.append(list(t))
    ll = []
    for lis in l:
        lll = []
        lll.extend(list(lis[0]))
        lll.append(lis[1])
        lll.extend(list(lis[2]))
        ll.append(lll)
    return ll
