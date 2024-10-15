import numpy as np
from tqdm import tqdm
import os
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# Contains functions to create the spectrograms of the stimuli and plot them 

def plot_spectrogram(times, frequencies, spectrogram, scaling="decibels", yscale="log", 
                     show=True, save=False, img_format=".png"):
    """Plots a spectrogram. Note that the spectrograms created using create_spectrograms are
    already on log-scale so linear scaling to decibels should not be used.
    """
        fig, ax = plt.subplots(1,1,figsize=(10,10))
    
        if scaling == "decibels":  
            im = ax.pcolormesh(times, frequencies, spectrogram,
                               shading='auto', norm=colors.LogNorm(vmin=spectrogram.min(), vmax=spectrogram.max()))
        elif scaling != "decibels":
            im = ax.pcolormesh(times, frequencies, spectrogram,
                               shading='auto')
            
        ax.set_yscale(yscale)
        ax.set_ylim(min(frequencies),max(frequencies))
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax) 
        if show == True:
            plt.show()
        
        if type(save) == str:
            fig.suptitle(save, fontsize=16)
            fig.savefig(save+img_format)
        
            
        #Alternative plotting function
        #Pxx, freqs, bins, im = ax.specgram(samples, NFFT=2940, Fs=44100, noverlap=0)

def create_spectrograms(fmin=10, fmax=4000, FFT_width=int(44100/15), FFT_overlap=int(44100/30), 
                        downsample=2, plot=False, scaling="decibels"):
    """Returns dictionary of spectrograms of all stimuli (even unused) and pure tones 
    represented by 1 sec long spectrograms. Dictionary contains for each stimulus:
    time at middle of each time bin, frequencies, spectrogram, time at end of each time bin.
    fmin = minimum frequency
    fmax = maximum frequency
    FFT_width and overlap parameterize the scipy spectrograms
    spectrograms are downsampled along frequency axis by 2.
    decibels scaling returns log of original spectrograms
    
    """
    spectrograms = {}
    path = "auditorycoding/audiostimfiles/ComplexAudioStimFiles_03_2021/"
    stimulus_names = [file for file in os.listdir(path)
                     if os.path.isfile(os.path.join(path, file))]

    for i in tqdm(range(len(stimulus_names))):
        sample_rate, samples = wavfile.read(path+stimulus_names[i])
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=FFT_width, noverlap=FFT_overlap)
        endtimes = times + 1/30 # Adds half of the window width (in hz)
        freq_slice = np.where((frequencies >= fmin) & (frequencies <= fmax))
        # keep only frequencies of interest
        frequencies   = frequencies[freq_slice]
        spectrogram = spectrogram[freq_slice,:][0]
        
        spectrogram_downsampled = np.zeros(shape=(int(spectrogram.shape[0]/downsample), spectrogram.shape[1]))
        frequencies_downsampled = np.zeros(shape=(int(len(frequencies)/downsample)))
        for t in range(spectrogram.shape[1]):
            if (spectrogram[:,t] == 0).all():
                if t > 0:
                    spectrogram[:,t] = spectrogram[:,t - 1]
                elif t == 0:
                    for time in range(t, spectrogram.shape[1]):
                        if (spectrogram[:,time] != 0).any():
                            spectrogram[:,t] = spectrogram[:,time]
                            break
                    
            for f in range(0, spectrogram.shape[0], downsample): 
                frequencies_downsampled[int(f/downsample)] = np.mean(frequencies[f:f+downsample])
                spectrogram_downsampled[int(f/downsample),t] = np.mean(spectrogram[f:f+downsample,t])
        if plot == True:
            #plot_spectrogram(times, frequencies, spectrogram)
            plot_spectrogram(times, frequencies_downsampled, spectrogram_downsampled, scaling=scaling)
        #print(spectrogram)
        #print(spectrogram_downsampled)
        #for j in range(1,len(times)):
            #print(times[j]-times[j-1])
        
        if scaling == "decibels":
            spectrograms[i] = [times, frequencies_downsampled, 10*np.log10(spectrogram_downsampled), endtimes]
        else:
            spectrograms[i] = [times, frequencies_downsampled, spectrogram_downsampled, endtimes]

    # Added the pure tone spectrograms behind the complex ones 

    sample_rate, samples = wavfile.read(path+stimulus_names[0])
    frequencies, times, _ = signal.spectrogram(samples, sample_rate, nperseg=FFT_width, noverlap=FFT_overlap)
    endtimes = times + 1/30 # Adds half of the window width (in hz)
    endtimes = endtimes[endtimes <= 1]
    times = times[:len(endtimes)]
    freq_slice = np.where((frequencies >= fmin) & (frequencies <= fmax))
    # keep only frequencies of interest
    frequencies = frequencies[freq_slice]
    
    frequencies_downsampled = np.zeros(shape=(int(len(frequencies)/downsample)))
    for t in range(spectrogram.shape[1]):          
        for f in range(0, spectrogram.shape[0], downsample): 
            frequencies_downsampled[int(f/downsample)] = np.mean(frequencies[f:f+downsample])

    max_amp = 73.37473957167927
    # Note that the last three frequencies are not working since they are out of bounds
    pure_frequencies = {1: 100.0, 2: 156.71210662, 3: 245.58684361, 4: 384.86431619, 5: 603.12897753, 
                        6: 945.17612631, 7: 1481.20541879, 8: 2321.22821513, 9: 3637.64563534,
                        10: 5700.63110645,11: 8933.57909744,12: 14000.}
   
    for i in pure_frequencies:
        frequency_bin = np.argmax(frequencies_downsampled[frequencies_downsampled < pure_frequencies[i]])
        spectrogram = np.zeros(shape=(len(frequencies_downsampled), len(endtimes)))
        spectrogram[frequency_bin,:] = max_amp
        spectrograms[i + len(stimulus_names) -1] = [times, frequencies_downsampled, spectrogram, endtimes]
    return spectrograms