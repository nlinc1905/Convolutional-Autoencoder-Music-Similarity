'''
This script pre-processes the MP3 data for autoencoding.
The MP3 files in a specified directory are first converted to wav files, using
the FFMPEG converter, which must be downloaded and installed from here:
https://ffmpeg.org/
Keep in mind that the MP3 to wav conversion is not lossless.

The output of this script is a csv file with the metadata for each song, along
with the wav file for each song.
'''

import os
import glob
from mutagen.id3 import ID3
import csv
from pydub import AudioSegment

#Specify a file directory and the types of audio files to convert to wav
filedir = 'C:/Users/Public/Documents/Python Scripts/Music Recommendation with Deep Learning/Audio Files/'
extension_list = ('*.mp3')

#Tell pydub where to find the ffmpeg audio converter
AudioSegment.converter = "C:/Program Files (x86)/FFMPEG/bin/ffmpeg.exe"

def getTags(path):
    """
    This function takes an mp3 filepath as input.  By default, Mutagen's ID3
    function updates tags to the latest version of ID3.
    
    The function outputs a dictionary of tags.
    If a tag is not found in the mp3's metadata, the output will be blank.
    """
    audio = ID3(path)
    audiotags = {}
    try:
        audiotags['artist'] = audio['TPE1'].text[0]
    except:
        audiotags['artist'] = None
    try:
        audiotags['song'] = audio['TIT2'].text[0]
    except:
        audiotags['song'] = None
    try:
        audiotags['album'] = audio['TALB'].text[0]
    except:
        audiotags['album'] = None
    try:
        audiotags['year'] = audio['TDRC'].text[0]
    except:
        audiotags['year'] = None
    return audiotags

#Iterate through the mp3's in the directory, compile a list of tags and convert the files to wav
os.chdir(filedir)
tagslist = []
id_tracker = 1
for extension in extension_list:
    for file in glob.glob(extension):
        if (os.path.splitext(os.path.basename(file))[1] == '.mp3'):
            print(file)
            mp3tags = getTags(file)
            mp3tags['audio_file_id'] = id_tracker
            tagslist.append(mp3tags)
            id_tracker = id_tracker + 1
            audio = AudioSegment.from_mp3(file)
            audio.export(os.path.splitext(os.path.basename(file))[0]+".wav", format="wav")

#Write the list of dictionaries with song metadata to a csv file
with open('Song_Metadata.csv', 'w') as f:
    w = csv.DictWriter(f, tagslist[0].keys())
    w.writeheader()
    w.writerows(tagslist)
