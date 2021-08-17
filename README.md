# AE-ML_Framework
Code for unsupervised learning of AE signals of SiC/SiC miniocomposites

ae_measure2.py file contains sub-routines for reading in AE data generated from WaveExplorer software and conversion of waves to feature vectors.

cluster_waves.py is the script which reads in an AE experiment, reads in a list of suitable waveforms, filters out the unsuitable waveforms, converts the suitable
waveforms to feature vectors, and clusters the feature vectors. The output is a csv of each event and its corresponding label.

In each experiment, Channel A/B/C/D corresponds to sensor S9225a, S9225-b, B1025-a, B1025-b respectively.

Raw waveforms and filters are included here. If there are questions as to how to import them, please contact me!



