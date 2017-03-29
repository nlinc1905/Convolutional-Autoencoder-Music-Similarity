'''
This script pre-processes the MP3 data for autoencoding.
Several features are calculated for wav files in a specified directory, 
which have been converted to wav from MP3 format.

The output of this script is Mel Spectrogram images for each wav file.
'''

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import glob
import csv
import math

#Define all major scales to be used later for finding key signature
#Arrays all in the format:  [C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B]
majorscales = {'C' : [1,0,1,0,1,1,0,1,0,1,0,1],
               'C#': [1,1,0,1,0,1,1,0,1,0,1,0],
               'D' : [0,1,1,0,1,0,1,1,0,1,0,1],
               'Eb': [1,0,1,1,0,1,0,1,1,0,1,0],
               'E' : [0,1,0,1,1,0,1,0,1,1,0,1],
               'F' : [1,0,1,0,1,1,0,1,0,1,1,0],
               'F#': [0,1,0,1,0,1,1,0,1,0,1,1],
               'G' : [1,0,1,0,1,0,1,1,0,1,0,1],
               'Ab': [1,1,0,1,0,1,0,1,1,0,1,0],
               'A' : [0,1,1,0,1,0,1,0,1,1,0,1],
               'Bb': [1,0,1,1,0,1,0,1,0,1,1,0],
               'B' : [0,1,0,1,1,0,1,0,1,0,1,1]}

class Audio(object):
    
    """
    Song objects are initiated with librosa.load() which produces an array 
    containing wav data in the first index and the wav's sample frequency 
    in the second.
    
    Stereo audio will be converted to mono by librosa.load() by averaging 
    the left and right channels.  This halves both the sample frequency and 
    the number of sample points. Note that the channel averaging method of 
    conversion gives each channel equal weight, which may not always be 
    appropriate. Lossless conversion of stereo to mono is impossible.  
    
    Instead of converting to mono, file could be imported as stereo and each 
    channel could be accessed individually by setting mono=False and subsetting:
    wav[:,0] and wav[:,1]
    
    wav.dtype will be 1 of 2 types:
        1) 16-bit - This means that the sound pressure values are mapped to 
        integer values ranging from -2^15 to (2^15)-1.  If wav.dtype is 16-bit,
        it will need to be converted to 32-bit ranging from -1 to 1
        2) 32-bit - This means that the sound pressure values are mapped to
        floating point values ranging from -1 to 1
   """
    
    def __init__(self, loadedAudio):
        self.wav = loadedAudio[0]
        self.samplefreq = loadedAudio[1]
        #If imported as 16-bit, convert to floating 32-bit ranging from -1 to 1
        if (self.wav.dtype == 'int16'):
            self.wav = self.wav/(2.0**15)
        self.channels = 1  #Assumes mono, if stereo then 2 (found by self.wav.shape[1])
        self.sample_points = self.wav.shape[0]
        self.audio_length_seconds = self.sample_points/self.samplefreq
        self.time_array_seconds = np.arange(0, self.sample_points, 1)/self.samplefreq
        self.tempo_bpm = librosa.beat.beat_track(y=self.wav, sr=self.samplefreq)[0]
        self.beat_frames = librosa.beat.beat_track(y=self.wav, sr=self.samplefreq)[1]
        #Transform beat array into seconds (these are the times when the beat hits)
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.samplefreq)
        #Get the rolloff frequency - the frequency at which the loudness drops off by 90%, like a low pass filter
        self.rolloff_freq = np.mean(librosa.feature.spectral_rolloff(y=self.wav, sr=self.samplefreq, hop_length=512, roll_percent=0.9))
     
    def plotWav(self):
        plt.plot(self.time_array_seconds, self.wav, color='k')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.show()

    def getTempo(self):
        print('Estimated tempo: {:.2f} beats per minute'.format(self.tempo_bpm))
    
    def getPercussiveTempo(self):
        #Separate the harmonics and percussives into 2 waves
        wav_harm, wav_perc = librosa.effects.hpss(self.wav)
        #Beat track the percussive signal
        tempo, beat_frames = librosa.beat.beat_track(y=wav_perc, sr=self.samplefreq)
        print('Estimated percussive tempo: {:.2f} beats per minute'.format(tempo))
        return tempo
    
    def getZeroCrossingRates(self):
        """
        ZCR is the count of times signal crosses 0 in a wave.  It is useful 
        for speech recognition and separating speech from background noise.
        ZCR will be smaller when a voice is speaking (0 is crossed less 
        frequently) and larger when there is a lot of background noise (0 is 
        crossed more frequently)
        
        ZCR is calculated by frame
        """
        zcrs = librosa.feature.zero_crossing_rate(y=self.wav, frame_length=2048, hop_length=512)
        return zcrs
        
    def plotChromagram(self):
        #Get chromagram of frequencies
        chroma = librosa.feature.chroma_stft(y=self.wav, sr=self.samplefreq)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        plt.show()
        return chroma
        
    def plotSpectrogram(self, mels=512, maxfreq=30000):
        #Plot the Mel power-scaled frequency spectrum, with any factor of 128 frequency bins and 512 frames (frame default)
        mel = librosa.feature.melspectrogram(y=self.wav, sr=self.samplefreq, n_mels=mels, fmax=maxfreq)
        librosa.display.specshow(librosa.logamplitude(mel, ref_power=np.max), y_axis='mel', fmax=maxfreq, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Power-Scaled Frequency Spectrogram')
        plt.tight_layout()
        plt.show()
        return mel   
    
    def plotMFCCs(self):
        """
        The Mel Frequency Cepstral Coefficient is a measure of timbre
        """
        mfccs = librosa.feature.mfcc(y=self.wav, sr=self.samplefreq)
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()
        return mfccs
    
    def plotTempogram(self):
        """
        The tempogram visualizes the rhythm (pattern recurrence), using the 
        onset envelope, oenv, to determine the start points for the patterns.
        """
        oenv = librosa.onset.onset_strength(y=self.wav, sr=self.samplefreq, hop_length=512)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.samplefreq, hop_length=512)
        librosa.display.specshow(tempogram, sr=self.samplefreq, hop_length=512, x_axis='time', y_axis='tempo')
        plt.colorbar()
        plt.title('Tempogram')
        plt.tight_layout()
        plt.show()
        plt.plot(oenv, label='Onset strength')
        plt.title('Onset Strength Over Time')
        plt.xlabel('Time')
        plt.ylabel('Onset Strength')
        plt.show()
        return tempogram

    def findTonicAndKey(self):
        """
        The tonic is the base note in the key signature, e.g. c is the tonic for 
        the key of c major.  The tonic can be found by summing the chromagram
        arrays and finding the index of the array with the greatest sum.  The 
        logic is that the tonic is the note with the greatest presence.
        
        If the tonic doesn't match the tonic of bestmatch, the highest 
        correlated major scale, then the key is a minor scale. 
        (Minor scales = Major scales but have different tonics)
        """
        chromagram = librosa.feature.chroma_stft(y=self.wav, sr=self.samplefreq)
        chromasums = []
        for i,a in enumerate(chromagram):
            chromasums.append(np.sum(chromagram[i]))
        tonicval = np.where(max(chromasums)==chromasums)[0][0]
        notes = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        tonic = notes[tonicval]
        #In standard units, how far is the average pitch from the tonic?
        z_dist_avg_to_tonic = round((max(chromasums)-np.mean(chromasums))/np.std(chromasums), 4)
        #Correlate the chromasums array with each of the major scales, find the best match
        bestmatch = 0
        bestmatchid = 0
        for key, scale in majorscales.items():
            #np.corrcoef returns a matrix, only need the first value in the diagonal
            corr = np.corrcoef(scale, chromasums)[0,1]
            if (corr > bestmatch):
                bestmatch = corr
                bestmatchid = key
        if (tonic != bestmatchid):
            keysig = tonic + ' Minor'
        else:
            keysig = tonic + ' Major'        
        return tonic, keysig, z_dist_avg_to_tonic
    
#Specify a file directory and the types of audio files to get features for
filedir = 'C:/Users/Public/Documents/Python Scripts/Music Recommendation with Deep Learning/Audio Files/'
extension_list = ('*.wav')

#Iterate through the wavs in the directory and compile a list of features
os.chdir(filedir)
featurelist = []
melspecs = []
id_tracker = 1
for extension in extension_list:
    for file in glob.glob(extension):
        if (os.path.splitext(os.path.basename(file))[1] == '.wav'):
            print(file)
            song = Audio(librosa.load(file, mono=True))
            wavfeatures = dict()
            wavmel = dict()
            wavfeatures['audio_file_id']        = id_tracker
            wavfeatures['samplefreq']           = song.samplefreq
            wavfeatures['channels']             = song.channels
            wavfeatures['sample_points']        = song.sample_points
            wavfeatures['audio_length_seconds'] = round(song.audio_length_seconds, 4)
            wavfeatures['tempo_bpm']            = song.tempo_bpm
            wavfeatures['avg_diff_beat_times']  = round(np.mean(song.beat_times[1:]-song.beat_times[0:len(song.beat_times)-1]), 4)
            wavfeatures['std_diff_beat_times']  = round(np.std(song.beat_times[1:]-song.beat_times[0:len(song.beat_times)-1]), 4)
            wavfeatures['rolloff_freq']         = round(song.rolloff_freq, 0)
            wavfeatures['avg_zcr']              = round(np.mean(song.getZeroCrossingRates()), 4)
            wavfeatures['zcr_range']            = np.max(song.getZeroCrossingRates()) - np.min(song.getZeroCrossingRates())
            wavfeatures['avg_mel_freq']         = round(np.mean(song.plotSpectrogram()), 4)
            wavfeatures['std_mel_freq']         = round(np.std(song.plotSpectrogram()), 4)
            wavfeatures['avg_onset_strength']   = round(np.mean(song.plotTempogram()), 4)
            wavfeatures['std_onset_strength']   = round(np.std(song.plotTempogram()), 4)
            wavfeatures['tonic']                = song.findTonicAndKey()[0]
            wavfeatures['key_signature']        = song.findTonicAndKey()[1]
            wavfeatures['z_dist_avg_to_tonic']  = song.findTonicAndKey()[2]
            wavmel['audio_file_id']             = id_tracker
            #wavmel['mel_spectrogram_sample']    = (song.plotSpectrogram(mels=512, maxfreq=8192)).ravel()[song.samplefreq*30:song.samplefreq*90]
            startcol = math.ceil((song.samplefreq*30)/512)
            endcol = math.ceil((song.samplefreq*90)/512)
            wavmel['mel_spectrogram_sample']    = (song.plotSpectrogram(mels=512, maxfreq=8192))[:, startcol:endcol]
            featurelist.append(wavfeatures)
            melspecs.append(wavmel)
            id_tracker = id_tracker + 1

#Write the list of dictionaries with song features to a csv file
with open('Song_Features.csv', 'w') as f:
    w = csv.DictWriter(f, featurelist[0].keys())
    w.writeheader()
    w.writerows(featurelist)

'''
Ideally the entire mel frequency spectrogram for each song would be exported, 
but the songs are all different lengths, meaning that the dimensions of the 
spectrograms will be different.  To standardize them all, I'm using 512 
frequency bins and taking a 60 second sample of each song.  I'm starting 30
seconds into the song to skip over any song intros and get into the main verse
and/or chorus.  

The spectrogram is clipped at a max of 8192 Hz, as there are few songs with 
higher frequencies present, so there is mostly black space above 8192 Hz.

Once the mel spectrogram is built, it is vectoriezed to a 1D array and then the 
subsetting is done.  The spectrogram is exported so that 1 song gets 1 file.
'''

#Specify a file directory for the spectrograms
specfiledir = 'C:/Users/Public/Documents/Python Scripts/Music Recommendation with Deep Learning/Audio Files/Spectrograms/'
if not os.path.exists(specfiledir):
    os.makedirs(specfiledir)
os.chdir(specfiledir)

#Export all spectorgrams to csv files
for d in melspecs:
    filename = str(d['audio_file_id']) + '.csv'
    print(filename)
    print(d['mel_spectrogram_sample'].shape)
    np.savetxt(filename, d['mel_spectrogram_sample'], delimiter=",")
    

