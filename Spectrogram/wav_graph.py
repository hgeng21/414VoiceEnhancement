import librosa
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import os


## This program reads the input '.wav' file and outputs the spectrogram in respect to the sound signal
def read_audio_spectum(filename):
    N_FFT=2048
    x, fs = librosa.load(filename, duration=58.04) # Duration=58.05 so as to make sizes convenient
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))  
    return S, fs


## This part is for outputing the spectrogram in respect to the 4 give sample sound signal
def output_orig_spectro(aud_samp1_1, aud_samp1_2, aud_samp2_1, aud_samp2_2):

    
    #aud_samp1_1 = "032.m4a.wav"
    #aud_samp1_2 = "037.m4a.wav"
    #aud_samp2_1 = "141.m4a.wav"
    #aud_samp2_2 = "147.m4a.wav"

    aud_samp1_1_audio, aud_samp1_1_sr = read_audio_spectum(aud_samp1_2)
    aud_samp1_2_audio, aud_samp1_2_sr = read_audio_spectum(aud_samp1_1)
    aud_samp2_1_audio, aud_samp2_1_sr = read_audio_spectum(aud_samp2_1)
    aud_samp2_2_audio, aud_samp2_2_sr = read_audio_spectum(aud_samp2_2)

    print(aud_samp1_1_audio.shape)
    print(aud_samp1_2_audio.shape)
    print(aud_samp2_1_audio.shape)
    print(aud_samp2_2_audio.shape)

    plt.figure(figsize=(10,7))
    plt.subplot(2,2,1)
    #plt.title('Audio Sample 1_1')
    plt.title(aud_samp1_1)
    plt.imshow(aud_samp1_2_audio[:500,:500])
    plt.subplot(2,2,2)
    #plt.title('Audio Sample 1_2')
    plt.title(aud_samp1_2)
    plt.imshow(aud_samp1_1_audio[:500,:500])
    plt.subplot(2,2,3)
    #plt.title('Audio Sample 2_1')
    plt.title(aud_samp2_1)
    plt.imshow(aud_samp2_1_audio[:500,:500])
    plt.subplot(2,2,4)
    #plt.title('Audio Sample 2_2')
    plt.title(aud_samp2_2)
    plt.imshow(aud_samp2_2_audio[:500,:500])
    plt.show()

#########


def add_noise(infile):
    # Import the pysndfx package and create an audio effects chain function.
    # Install from shell
    # Command: pip install pysndfx
    # Reference: https://pypi.org/project/pysndfx/
    from pysndfx import AudioEffectsChain

    fx = (
        AudioEffectsChain()
        .highshelf()
        .reverb()
        .phaser()
        .delay()
        .lowshelf()
    )

    fx = (
        AudioEffectsChain()
        .highshelf()
        .reverb()
        .phaser()
        .delay()
        .lowshelf()
    )

    highshelf = (
        AudioEffectsChain()
        .highshelf()
    )

    lowshelf = (
        AudioEffectsChain()
        .lowshelf()
    )

    HLshelf = (
        AudioEffectsChain()
        .highshelf()
        .lowshelf()
    )


    reverb = (
        AudioEffectsChain()
        .reverb()
    )

    phaser = (
        AudioEffectsChain()
        .phaser()
    )

    delay = (
        AudioEffectsChain()
        .delay()
    )

    #infile = "032.m4a.wav"
    #outfile = "032_noise_added.wav"

    outfile = infile[0:-4] + "_noise_added.wav"
    outfile_highshelf = infile[0:-4] + "_noise_added_highshelf.wav"
    outfile_lowshelf = infile[0:-4] + "_noise_added_lowshelf.wav"
    outfile_HLshelf = infile[0:-4] + "_noise_added_HLshelf.wav"
    outfile_reverb = infile[0:-4] + "_noise_added_reverb.wav"
    outfile_phaser = infile[0:-4] + "_noise_added_phaser.wav"
    outfile_delay = infile[0:-4] + "_noise_added_delay.wav"
    outfile_phaser_delay = infile[0:-4] + "_noise_added_phaser_delay.wav"

    ############# Apply phaser and reverb directly to an audio file #############
    fx(infile, outfile_phaser_delay)

    ###################### Apply effects ######################
    y, sr = librosa.load(infile, sr=None)
    y = fx(y)
    # Apply the effects and return the results as a ndarray.
    y = fx(infile)
    # Apply the effects to a ndarray but store the resulting audio to disk.
    fx(y, outfile)

    ###################### Apply highshelf effects ######################
    # Apply the effects to a ndarray but store the resulting audio to disk.
    highshelf(y, outfile_highshelf)
    
    ###################### Apply lowshelf effects ######################
    # Apply the effects to a ndarray but store the resulting audio to disk.
    lowshelf(y, outfile_lowshelf)

    ###################### Apply high and lowshelf effects ######################
    # Apply the effects to a ndarray but store the resulting audio to disk.
    HLshelf(y, outfile_HLshelf)
    
    ###################### Apply reverb effects ######################
    # Apply the effects to a ndarray but store the resulting audio to disk.
    reverb(y, outfile_reverb)
    
    ###################### Apply phaser effects ######################
    # Apply the effects to a ndarray but store the resulting audio to disk.
    phaser(y, outfile_phaser)
    
    ###################### Apply delay effects ######################
    # Apply the effects to a ndarray but store the resulting audio to disk.
    delay(y, outfile_delay)



if __name__ == "__main__":
    aud_samp1_1 = "032.m4a.wav"
    aud_samp1_2 = "037.m4a.wav"
    aud_samp2_1 = "141.m4a.wav"
    aud_samp2_2 = "147.m4a.wav"

    # Outputing the spectrogram in respect to the 4 give sample sound signal
    output_orig_spectro(aud_samp1_1, aud_samp1_2, aud_samp2_1, aud_samp2_2)

    noised1_1 = aud_samp1_1[0:-4] + "_noise_added.wav"
    noised1_1_highshelf = aud_samp1_1[0:-4] + "_noise_added_highshelf.wav"
    noised1_1_lowshelf = aud_samp1_1[0:-4] + "_noise_added_lowshelf.wav"
    noised1_1_HLshelf = aud_samp1_1[0:-4] + "_noise_added_HLshelf.wav"
    noised1_1_reverb = aud_samp1_1[0:-4] + "_noise_added_reverb.wav"
    noised1_1_phaser = aud_samp1_1[0:-4] + "_noise_added_phaser.wav"
    noised1_1_delay = aud_samp1_1[0:-4] + "_noise_added_delay.wav"
    noised1_1_phaser_delay = aud_samp1_1[0:-4] + "_noise_added_phaser_delay.wav"

    noised2_1 = aud_samp2_1[0:-4] + "_noise_added.wav"
    noised2_1_highshelf = aud_samp2_1[0:-4] + "_noise_added_highshelf.wav"
    noised2_1_lowshelf = aud_samp2_1[0:-4] + "_noise_added_lowshelf.wav"
    noised2_1_HLshelf = aud_samp2_1[0:-4] + "_noise_added_HLshelf.wav"
    noised2_1_reverb = aud_samp2_1[0:-4] + "_noise_added_reverb.wav"
    noised2_1_phaser = aud_samp2_1[0:-4] + "_noise_added_phaser.wav"
    noised2_1_delay = aud_samp2_1[0:-4] + "_noise_added_delay.wav"
    noised2_1_phaser_delay = aud_samp2_1[0:-4] + "_noise_added_phaser_delay.wav"

    add_noise(aud_samp1_1)
    add_noise(aud_samp2_1)
    
    # Outputing the spectrogram of the original sound and highshelf, noise effects added sound
    output_orig_spectro(aud_samp1_1, aud_samp2_1, noised1_1_highshelf, noised2_1_highshelf)
    
    # Outputing the spectrogram of the original sound and lowshelf noise effects added sound
    output_orig_spectro(aud_samp1_1, aud_samp2_1, noised1_1_lowshelf, noised2_1_lowshelf)

    # Outputing the spectrogram of the original sound and (high and lowshelf) noise effects added sound
    output_orig_spectro(aud_samp1_1, aud_samp2_1, noised1_1_HLshelf, noised2_1_HLshelf)
    
    # Outputing the spectrogram of the original sound and reverb noise effects added sound
    output_orig_spectro(aud_samp1_1, aud_samp2_1, noised1_1_reverb, noised2_1_reverb)
    
    # Outputing the spectrogram of the original sound and phaser noise effects added sound
    output_orig_spectro(aud_samp1_1, aud_samp2_1, noised1_1_phaser, noised2_1_phaser)
    
    # Outputing the spectrogram of the original sound and delay noise effects added sound
    output_orig_spectro(aud_samp1_1, aud_samp2_1, noised1_1_delay, noised2_1_delay)
    
    # Outputing the spectrogram of the original sound and (phaser and delay) noise effects added sound
    output_orig_spectro(aud_samp1_1, aud_samp2_1, noised1_1_phaser_delay, noised2_1_phaser_delay)

    # Outputing the spectrogram of the original sound and all noise effects added sound
    output_orig_spectro(aud_samp1_1, aud_samp2_1, noised1_1, noised2_1)

''' From the output comparation spectogram of the original voice spectogram and the noice-added spectogram we can see that
    
'''
    




















