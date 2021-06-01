'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 210518

This script takes in a file path with one experiment as text file
It clusters with k=2


'''

#Imports
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
import glob
import sklearn
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
from scipy.cluster.vq import whiten
import pandas as pd
import os
from scipy.cluster.vq import whiten
from scipy.integrate import simps



def get_match_rate(label_1, label_2):
    acc = 0
    for i, label in enumerate(label_1):
        if label == label_2[i]:
            acc = acc+1
    acc = acc/len(label_1)
    if acc <.5:
        return 1-acc
    else:
        return acc



if __name__ == "__main__":

    '''
    Read-in and Setup
    '''
    sig_len = 1024

    fname_raw = '210308-1_waveforms'
    fname_filter = '210308-1_filter_agg'

    raw = glob.glob("./Raw_Data/2_Sensor/210308-1/"+fname_raw+".txt")[0]
    filter = glob.glob("./Filtered_Data/2_Sensor/210308-1/"+fname_filter+".csv")[0]

    csv = pd.read_csv(filter)

    time = np.array(csv.Time)
    stress = np.array(csv.Adjusted_Stress_MPa)


    v0, ev = filter_ae(raw, filter, channel_num=0, sig_length=sig_len) # S9225
    v1, ev = filter_ae(raw, filter, channel_num=1, sig_length=sig_len) # S9225
    v2, ev = filter_ae(raw, filter, channel_num=2, sig_length=sig_len) # B1025
    v3, ev = filter_ae(raw, filter, channel_num=3, sig_length=sig_len) # B1025


    channels = [v0, v1, v2, v3]
    #channels = [v0, v1, v2]


    '''
    Parameters
    '''
    k = 2
    NN = 5

    dt = 10**-7 #s
    Low = 200*10**3 #Hz
    High = 800*10**3 #Hz

    num_bins = 26



    FFT_units = 1000 #FFT outputs in Hz, this converts to kHz

    spectral = SpectralClustering(n_clusters=k, n_init=100, eigen_solver='arpack'
                                    ,affinity="nearest_neighbors",  n_neighbors=NN)
    '''
    Cast experiment as vectors
    '''

    vect = []
    for channel in channels:
        channel_vector = []
        for waveform in channel:
            feature_vector, freq_bounds, spacing = wave2vec(dt, waveform, Low, High, num_bins, FFT_units)
            channel_vector.append(feature_vector) # set of all waveforms from channel as a vector
        vect.append(channel_vector) # set of all waveforms from experiment as vector index: i,j,k -> channel, waveform #, vector entry

    # Cluster waveform3
    ch0_X = vect[0]
    ch1_X = vect[1]
    ch2_X = vect[2]
    ch3_X = vect[3]



    # Cluster waveform
    spectral_A = spectral.fit(ch0_X)
    A_lads = spectral_A.labels_

    # Cluster waveform
    spectral_B = spectral.fit(ch1_X)
    B_lads = spectral_B.labels_

    # Cluster waveform
    spectral_C = spectral.fit(ch2_X)
    C_lads = spectral_C.labels_

    # Cluster waveform
    spectral_D = spectral.fit(ch3_X)
    D_lads = spectral_D.labels_








    '''
    #Channel A
    '''

    # Plotting routine for Danny_dont_vito
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
    ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.grid()

    plot1 = ax1.scatter(stress, A_lads+1 , color=color1, linewidth=width) #plot silh
    pl.title('Channel A', fontsize=BIGGER_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.savefig('Channel_A_mask.png')

    df = pd.DataFrame({'Event': ev,'Cluster': A_lads, 'Time': time, 'Stress': stress})
    df.to_csv(r'Channel_A_mask.csv')

    pl.clf()



    '''
    # Channel B mask
    '''



    # Plotting routine for Danny_dont_vito

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Cluster number', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.grid()

    plot1 = ax1.scatter(stress, B_lads+1 , color=color1, linewidth=width) #plot silh
    pl.title('Channel B', fontsize=BIGGER_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.savefig('channel_B_mask.png')


    df2 = pd.DataFrame({'Event': ev,'Cluster': B_lads, 'Time': time, 'Stress': stress})
    df2.to_csv(r'Channel_B_mask.csv')

    pl.clf()



    '''
    #Channel C
    '''

    # Plotting routine for Danny_dont_vito

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Cluster number', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.grid()

    plot1 = ax1.scatter(stress, C_lads+1 , color=color1, linewidth=width) #plot silh
    pl.title('Channel C', fontsize=BIGGER_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.savefig('Channel_C_mask.png')

    df3 = pd.DataFrame({'Event': ev,'Cluster': C_lads, 'Time': time, 'Stress': stress})
    df3.to_csv(r'Channel_C_mask.csv')

    pl.clf()



    '''
    # Channel D mask
    '''



    # Plotting routine for Danny_dont_vito

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Cluster number', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.grid()

    plot1 = ax1.scatter(stress, D_lads+1 , color=color1, linewidth=width) #plot silh
    pl.title('Channel D', fontsize=BIGGER_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.savefig('channel_D_mask.png')



    df4 = pd.DataFrame({'Event': ev,'Cluster': D_lads, 'Time': time, 'Stress': stress})
    df4.to_csv(r'Channel_D_mask.csv')

    pl.clf()


    print('Bin spacing (kHz): ' , spacing)
    # print('Frequency resolution (kHz): ' , dw)

    print('')

    print('S9225 ARI: ', ari(A_lads,B_lads))
    print('B1025 ARI: ', ari(C_lads, D_lads))
    print('Left ARI: ', ari(A_lads,C_lads))
    print('Right ARI: ', ari(B_lads, D_lads))
    print('A/D ARI: ' , ari(A_lads,D_lads))
    print('B/C ARI: ' , ari(B_lads,C_lads))

    print('')

    print('S9225 matching rate: ' , get_match_rate(A_lads, B_lads))
    print('B1025 matching rate: ' , get_match_rate(C_lads, D_lads))
    print('Left matching rate: ' , get_match_rate(A_lads, C_lads))
    print('Right matching rate: ' , get_match_rate(B_lads, D_lads))
    print('A/D matching rate: ' , get_match_rate(A_lads, D_lads))
    print('B/C matching rate: ' , get_match_rate(B_lads, C_lads))



    # NOTE: get error as function of damage parameter
'''
    match_rate = []
    for i in range(len(D_lads)-1):
        match_rate.append(ari(D_lads[0:i+1], C_lads[0:i+1]))
'''

'''
    # Plotting routine for Danny_dont_vito
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    width = 2.0

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Adjusted Rand Index', fontsize=MEDIUM_SIZE)
    #ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax1.set_xlabel('Event Number', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelsize=MEDIUM_SIZE)
    #ax1.grid()

    plot1 = ax1.plot(match_rate, linewidth=width)


    pl.title(fname_raw, fontsize=BIGGER_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.show()
'''
