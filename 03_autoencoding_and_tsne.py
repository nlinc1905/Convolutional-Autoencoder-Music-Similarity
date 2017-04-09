'''
This script trains an autoencoder on every csv file containing the Mel
spectrogram data for a list of songs. The learned latent features are then
used to cluster the songs with t-SNE

Artist information will be used to color the clusters to see how accurate 
clustering is.
'''

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adadelta
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

#Set seed for repeatability
seed = 14
np.random.seed(seed)

'''
-------------------------------------------------------------------------------
--------------------Convolutional Autoencoder Indv Model-----------------------
-------------------------------------------------------------------------------
'''

#Specify the file directory for the spectrogram csv files
filedir = '/home/nick/Documents/Python Scripts/Music Recommendation with Deep Learning/Audio Files/Spectrograms/'
os.chdir(filedir)


def autoencode(fileNbr):
    '''
    This function loads a csv file with the Mel spectrogram image data,
    then creates a convolutional autoencoder to learn latent fatures about the image.
    The output is the encoded image, reshaped to be 1 row.
    '''

    #Load data as array, noting that the log amplitude must be taken to scale the values
    spec = librosa.logamplitude(np.loadtxt(str(i) + '.csv', delimiter=','), ref_power=np.max)
    #Inspect the spectrogram. x-axis is time in frames and y-axis is frequency (Hz) upside down
    plt.imshow(spec)
    
    x_train = spec.astype('float32') / 255.
    x_train = np.reshape(x_train, (1, 512, 2584, 1))
    #Test data will be the same as training data
    
    input_img = Input(shape=(512, 2584, 1))
    
    #Build the endoder piece
    encoded = Conv2D(nb_filter=16, nb_row=3, nb_col=3, input_shape=(512, 2584, 1), activation='relu', border_mode='same')(input_img)
    encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
    encoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, activation='relu', border_mode='same')(encoded)
    encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
    encoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, activation='relu', border_mode='same')(encoded)
    encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
    encoder = Model(input=input_img, output=encoded)
    #encoded_imgs = encoder.predict(x_train)
    #plt.imshow(encoded_imgs[0].reshape(encoded_imgs[0].shape[0], encoded_imgs[0].shape[1]*encoded_imgs[0].shape[2]).T)
    
    #Build the decoder piece
    decoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, input_shape=(4, 4, 8), activation='relu', border_mode='same')(encoded)
    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, activation='relu', border_mode='same')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(nb_filter=16, nb_row=3, nb_col=3, activation='relu', border_mode='same')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(nb_filter=1, nb_row=3, nb_col=3, activation='sigmoid', border_mode='same')(decoded)
    #decoder = Model(input=encoder.input, output=decoded)
    #decoded_imgs = decoder.predict(x_train)
    #plt.imshow(decoded_imgs[0].reshape(decoded_imgs[0].shape[0], decoded_imgs[0].shape[1]))
    
    #Create learning rate schedule and add it to the optimizer of choice
    learning_rate = 1.0
    epochs = 50
    decay_rate = learning_rate / epochs
    #rho and epsilon left at defaults
    adadelta = Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=decay_rate)
    
    #Create a custom callback to stop training the autoencoder when a val_loss of 0.1 is reached
    class EarlyStoppingByLossVal(Callback):
        def __init__(self, monitor='val_loss', value=0.1, verbose=0):
            super(Callback, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose
    
        def on_epoch_end(self, epoch, logs={}):
            current = logs.get(self.monitor)
            if current < self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True
    
    #modelfilepath = "/home/nick/Documents/Python Scripts/Music Recommendation with Deep Learning/Models/" + str(i) + "autoencoder-{epoch:02d}-{val_loss:.2f}.hdf5"
    modelfilepath = "/home/nick/Documents/Python Scripts/Music Recommendation with Deep Learning/Models/" + str(i) + "autoencoder-best_fit.hdf5"
    callbacks = [
        EarlyStoppingByLossVal(monitor='val_loss', value=0.1, verbose=1),
        #Save the best model to a hdf5 file located at modelfilepath
        ModelCheckpoint(modelfilepath, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    
    #Build the full autoencoder
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=adadelta, loss='binary_crossentropy')
    autoencoder.summary()
    autoencoder.fit(x_train, x_train, nb_epoch=epochs, batch_size=128, shuffle=True,
                    validation_data=(x_train, x_train), callbacks=callbacks)
    
    #Reset the decoded images to be the result of the trained autoencoder
    autoencoded_imgs = librosa.logamplitude(autoencoder.predict(x_train), ref_power=np.max)
    plt.imshow(autoencoded_imgs[0].reshape(512, 2584))
    plt.imshow(x_train.reshape(512, 2584))
    
    #Store endoded image data for comparison to others.  (Encoder wieghts have been tuned now)
    encoded_imgs = encoder.predict(x_train)
    plt.imshow(encoded_imgs[0].reshape(encoded_imgs[0].shape[0], encoded_imgs[0].shape[1]*encoded_imgs[0].shape[2]).T)
    
    #Reshape each encoded image to be 1 row by 165,376 columns
    print(encoded_imgs.shape)
    print(encoded_imgs.reshape(encoded_imgs.shape[1], encoded_imgs.shape[2]*encoded_imgs.shape[3]).reshape(1, encoded_imgs.shape[1]*encoded_imgs.shape[2]*encoded_imgs.shape[3]).shape)
    enc_rs = encoded_imgs.reshape(encoded_imgs.shape[1], encoded_imgs.shape[2]*encoded_imgs.shape[3]).reshape(1, encoded_imgs.shape[1]*encoded_imgs.shape[2]*encoded_imgs.shape[3])

    return enc_rs[0]


#Build an autoencoder for every file in the directory (there are 28 files)
enc_list = []
for i in range(1, 29):
    enc_list.append(autoencode(i))
x = np.matrix(enc_list)
print(x.shape)
#Specify the song ID as the label y
y = np.array(range(1, 29))

'''
-------------------------------------------------------------------------------
------------------------Plotting Clusters Indv Model---------------------------
-------------------------------------------------------------------------------
'''

#Build TSNE model
tsne_model = TSNE(n_components=2, perplexity=2.0, learning_rate=100.0, 
                  n_iter=5000, n_iter_without_progress=60, 
                  random_state=seed, method='barnes_hut')
tsne_dim = tsne_model.fit_transform(x)

#Plot first 2 extracted features and the observation class
plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable x')
plt.ylabel('Latent Variable y')
plt.title('t-SNE 2-Dimension Plot with Observation Class')
#plt.axis([-0.0003, 0.0003, -0.0003, 0.0003])
plt.scatter(tsne_dim[:, 0], tsne_dim[:, 1], c=y, s=50)
plt.colorbar()
labels = ['{0}'.format(i) for i in y]
for label, xc, yc in zip(labels, tsne_dim[:, 0], tsne_dim[:, 1]):
    plt.annotate(
        label,
        xy=(xc, yc), xytext=(-0, 0),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

#Build K-Means clusters with k=8, based on looking at t-SNE results
km_model = KMeans(n_clusters=8)
km_results = km_model.fit(tsne_dim)

#Plot the clusters
plt.figure(figsize=(10,5))
plt.scatter(tsne_dim[:,0], tsne_dim[:,1], c=km_model.labels_, s=50)
plt.title('K Mean Clustering of t-SNE Reduced Dimensions')
for label, xc, yc in zip(labels, tsne_dim[:, 0], tsne_dim[:, 1]):
    plt.annotate(
        label,
        xy=(xc, yc), xytext=(-0, 0),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

'''
-------------------------------------------------------------------------------
-----------------------Convolutional Autoencoder-------------------------------
-------------------------------------------------------------------------------
'''

#Specify the file directory for the spectrogram csv files
filedir = '/home/nick/Documents/Python Scripts/Music Recommendation with Deep Learning/Audio Files/Spectrograms/'
os.chdir(filedir)

def readFile(filenbr):
    #Load data as array, noting that the log amplitude must be taken to scale the values
    spec = librosa.logamplitude(np.loadtxt(str(filenbr) + '.csv', delimiter=','), ref_power=np.max)
    x_train = spec.astype('float32') / 255.
    x_train = np.reshape(x_train, (512, 2584, 1))
    #Test data will be the same as training data
    return x_train

x_train = []
for i in range(1, 19):
    x_train.append(readFile(i))
x_train = np.array(x_train)
print(x_train.shape)

#Inspect a couple spectrograms (x-axis is time in frames and y-axis is freq upside down)
plt.title("Mel Spectrogram")
plt.xlabel("Time in Frames")
plt.ylabel("Mel-scaled Frequency (Hz)")
plt.imshow(x_train[0].reshape(512, 2584))
plt.title("Mel Spectrogram")
plt.xlabel("Time in Frames")
plt.ylabel("Mel-scaled Frequency (Hz)")
plt.imshow(x_train[1].reshape(512, 2584))

input_img = Input(shape=(512, 2584, 1))

#Build the endoder piece
encoded = Conv2D(nb_filter=16, nb_row=3, nb_col=3, input_shape=(512, 2584, 1), activation='relu', border_mode='same')(input_img)
encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
encoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, activation='relu', border_mode='same')(encoded)
encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
encoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, activation='relu', border_mode='same')(encoded)
encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
encoder = Model(input=input_img, output=encoded)
#encoded_imgs = encoder.predict(x_train)
#plt.imshow(encoded_imgs[0].reshape(encoded_imgs[0].shape[0], encoded_imgs[0].shape[1]*encoded_imgs[0].shape[2]).T)

#Build the decoder piece
decoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, input_shape=(4, 4, 8), activation='relu', border_mode='same')(encoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, activation='relu', border_mode='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(nb_filter=16, nb_row=3, nb_col=3, activation='relu', border_mode='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(nb_filter=1, nb_row=3, nb_col=3, activation='sigmoid', border_mode='same')(decoded)
#decoder = Model(input=encoder.input, output=decoded)
#decoded_imgs = decoder.predict(x_train)
#plt.imshow(decoded_imgs[0].reshape(decoded_imgs[0].shape[0], decoded_imgs[0].shape[1]))

#Create learning rate schedule and add it to the optimizer of choice
learning_rate = 1.0
epochs = 50
decay_rate = learning_rate / epochs
#rho and epsilon left at defaults
adadelta = Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=decay_rate)

#Create a custom callback to stop training the autoencoder when a val_loss of 0.1 is reached
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.1, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

modelfilepath = "/home/nick/Documents/Python Scripts/Music Recommendation with Deep Learning/Models/combined-data-autoencoder-best_fit.hdf5"
callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.1, verbose=1),
    #Save the best model to a hdf5 file located at modelfilepath
    ModelCheckpoint(modelfilepath, monitor='val_loss', save_best_only=True, verbose=0),
]

#Build the full autoencoder
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=adadelta, loss='binary_crossentropy')
autoencoder.summary()
autoencoder.fit(x_train, x_train, nb_epoch=epochs, batch_size=128, shuffle=True,
                validation_data=(x_train, x_train), callbacks=callbacks)

#Reset the decoded images to be the result of the trained autoencoder
autoencoded_imgs = librosa.logamplitude(autoencoder.predict(x_train), ref_power=np.max)
plt.imshow(autoencoded_imgs[0].reshape(512, 2584))
plt.imshow(x_train[0].reshape(512, 2584))

#Store endoded image data for comparison to others.  (Encoder wieghts have been tuned now)
encoded_imgs = encoder.predict(x_train)
plt.imshow(encoded_imgs[0].reshape(encoded_imgs[0].shape[0], encoded_imgs[0].shape[1]*encoded_imgs[0].shape[2]).T)

#Reshape each encoded images to be 18 rows by 64*323*8 columns
print(encoded_imgs.shape)
enc_rs = []
for ei in encoded_imgs:
    enc_rs.append(ei.reshape(1, ei.shape[0]*ei.shape[1]*ei.shape[2]))
x = np.array(enc_rs)

#Specify the song ID as the label y
y = np.array(range(1, 19))

'''
-------------------------------------------------------------------------------
----------------------------Plotting Clusters---------------------------------
-------------------------------------------------------------------------------
'''

#Build TSNE model
tsne_model = TSNE(n_components=2, perplexity=2.0, learning_rate=100.0, 
                  n_iter=5000, n_iter_without_progress=60, 
                  random_state=seed, method='barnes_hut')
tsne_dim = tsne_model.fit_transform(np.matrix(x))

#Plot first 2 extracted features and the observation class
plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable x')
plt.ylabel('Latent Variable y')
plt.title('t-SNE 2-Dimension Plot with Observation Class')
#plt.axis([-0.0003, 0.0003, -0.0003, 0.0003])
plt.scatter(tsne_dim[:, 0], tsne_dim[:, 1], c=y, s=50)
plt.colorbar()
labels = ['{0}'.format(i) for i in y]
for label, xc, yc in zip(labels, tsne_dim[:, 0], tsne_dim[:, 1]):
    plt.annotate(
        label,
        xy=(xc, yc), xytext=(-0, 0),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

#Build K-Means clusters with k=8, based on looking at t-SNE results
km_model = KMeans(n_clusters=8)
km_results = km_model.fit(tsne_dim)

#Plot the clusters
plt.figure(figsize=(10,5))
plt.scatter(tsne_dim[:,0], tsne_dim[:,1], c=km_model.labels_, s=50)
plt.title('K Mean Clustering of t-SNE Reduced Dimensions')
for label, xc, yc in zip(labels, tsne_dim[:, 0], tsne_dim[:, 1]):
    plt.annotate(
        label,
        xy=(xc, yc), xytext=(-0, 0),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

#Perform hierarchical agglomerative clustering
hc_model = linkage(tsne_dim, method='ward', metric='euclidean')

#Manually adding these since there aren't many
dendrogram_labels = ["Ain't No Grave - Johnny Cash",
                     "Danse Macabre - Camille Saint-Saens",
                     "Gimme Something Good - Ryan Adams",
                     "Mine - Taylor Swift",
                     "Sympohny No.3, Op.78 - Camille Saint-Saens",
                     "Roar - Katy Perry",
                     "Somewhat Damaged - Nine Inch Nails",
                     "Yesterday - The Beatles",
                     "Hurt - Johnny Cash",
                     "Shakermaker - Oasis",
                     "All My Loving - The Beatles",
                     "Back To December - Taylor Swift",
                     "Live Forever - Oasis",
                     "Wonderwall - Oasis",
                     "Don't Look Back In Anger - Oasis",
                     "Firework - Katy Perry",
                     "Little James - Oasis",
                     "Every Day Is Exactly The Same - Nine Inch Nails"]

#Plot the full dendrogram
plt.figure(figsize=(12, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Song')
plt.ylabel('Distance')
dendrogram(hc_model,
    leaf_rotation=90.,  #Rotate the x-axis labels for ease of reading
    leaf_font_size=10.,  #Specify font size of x-axis labels
    labels=dendrogram_labels
)
plt.show()