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
    sig_len = 10240

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







    channels = [v0]
    #channels = [v0, v1, v2]



    #Parameters

    k = 3
    NN = 100

    dt = 10**-7 #s
    Low = 000*10**3 #Hz
    High = 1000*10**3 #Hz

    num_bins = 5



    FFT_units = 1000 #FFT outputs in Hz, this converts to kHz

    spectral = SpectralClustering(n_clusters=k, n_init=100, eigen_solver='arpack'
                                    ,affinity="nearest_neighbors",  n_neighbors=NN)
    spectral_NN_10 = SpectralClustering(n_clusters=k, n_init=100, eigen_solver='arpack'
                                    ,affinity="nearest_neighbors",  n_neighbors=10)






    #Cast experiment as vectors
    print('Casting waves as vectors')
    vect = []
    for channel in channels:
        channel_vector = []
        for waveform in channel:
            feature_vector, freq_bounds, spacing = wave2vec(dt, waveform, Low, High, num_bins, FFT_units)
            channel_vector.append(feature_vector) # set of all waveforms from channel as a vector
        vect.append(channel_vector) # set of all waveforms from experiment as vector index: i,j,k -> channel, waveform #, vector entry

    # Cluster waveform
    ch0_X = vect[0]


    print('Begining clustering')
    # Cluster waveform
    spectral_A = spectral.fit(ch0_X)
    labels = spectral_A.labels_
    print('Clustering complete')




    os.chdir(write_directory)
    # Plotting routine
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    width = 2.0

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Cluster number', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.set_xlabel('Event number', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.grid()

    plot1 = ax1.scatter(ev, labels+1 , color=color1, linewidth=width) #plot silh
    pl.title('Clustred waveforms', fontsize=BIGGER_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.savefig('Clustered_QuartzAE.png')

    df = pd.DataFrame({'Event': ev,'Cluster': labels, 'Time': time})
    df.to_csv(r'clustered_waves.csv')

    pl.clf()



    print(labels)




    print('Bin spacing (kHz): ' , spacing)
    # print('Frequency resolution (kHz): ' , dw)

    print('')
