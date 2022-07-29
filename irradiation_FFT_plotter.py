'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 210518

This script takes in a file path with one experiment as text file
It clusters with k=2


'''

#Imports
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari
import sklearn
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
import pandas as pd

import os
from os import listdir, chdir
from os.path import isfile, join



if __name__ == "__main__":

    '''
    Read-in and Setup
    '''
    sig_len = 1024

    data_directory = 'C:/Research/Khalids_data/QuartzAE/'
    write_directory = 'C:/Research/Khalids_data/'
    os.chdir(data_directory)


    onlyfiles = [f for f in listdir(data_directory) if isfile(join(data_directory, f))] # NOTE: gets file list in directory
    v0, ev, time = [],[],[]

    print('Unpacking waveforms')
    for i, file in enumerate(onlyfiles):
        wave, num, hit_time = read_mistras(file)
        v0.append(wave)
        ev.append(num)
        time.append(hit_time)


    time, v0, ev = zip(*sorted(zip(time, v0, ev))) # NOTE: sorts the times and waveforms simultaneously
    time = np.array(time)
    v0 = np.array(v0)
    ev = np.array(ev)



    low = 10*10**3
    high = 1000*10**3
    dt=10**-7

    i = 1
    print(v0[i])
    pl.plot(v0[i])
    pl.show()
    w,z = fft(dt, v0[i], low_pass=low, high_pass=high)
    pl.plot(w/1000,z)
    pl.show()
