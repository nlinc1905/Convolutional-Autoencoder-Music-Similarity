'''
This script trains an autoencoder on every csv file containing the Mel
spectrogram data for a list of songs. The learned latent features are then
used to cluster the songs with t-SNE

Artist information will be used to color the clusters to see how accurate 
clustering is.
'''

