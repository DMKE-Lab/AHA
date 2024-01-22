import pickle
from dataset import BaseDataset
import numpy as np
import os
from tqdm import tqdm
import pandas
d_train = pandas.read_csv('data/GDELT/train.txt', sep='\t', header=None, names=['1', '2', '3', '4', '5'])
d_valid = pandas.read_csv('data/GDELT/valid.txt', sep='\t', header=None, names=['1', '2', '3', '4', '5'])
d_test = pandas.read_csv('data/GDELT/test.txt', sep='\t', header=None, names=['1', '2', '3', '4', '5'])
d_train['4'] = d_train['4']/60
d_valid['4'] = d_valid['4']/60
d_test['4'] = d_test['4']/60
d_train['4'] = d_train['4'].astype(int)
d_valid['4'] = d_valid['4'].astype(int)
d_test['4'] = d_test['4'].astype(int)
d_train.to_csv('train.txt', sep='\t', header=False, index=False)
d_valid.to_csv('test.txt', sep='\t', header=False, index=False)
d_test.to_csv('valid.txt', sep='\t', header=False, index=False)