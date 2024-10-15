import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy import stats

# Contains functions to preprocess data to generate datasets
# masking, cross-validation, matching of input and output

def generate_masks(relevance_cutoff):
    """Generates masks for widefield images of original size, downsampled, and downsampled twice.
    relevance_cutoff determines which pixels are considered as masked when downsampling.
    """
    roi_morphed = np.load(open("auditorycoding/F189/masks/1_roi_morphed.npy", "rb"))
    mask_stabilized = np.load(open("auditorycoding/F189/masks/mask_stabilized.npy", "rb"))   
    mask = roi_morphed * mask_stabilized[:,:,1] # Combine ROI and blood vessel mask
    factor = 2 # downsampling factor used in each iteration
    roi_downsampled = np.zeros(shape=(int(mask.shape[0]/factor), int(mask.shape[1]/factor)))
    mask_downsampled = np.zeros(shape=(int(mask.shape[0]/factor), int(mask.shape[1]/factor)))
    for y in range(0, mask.shape[0], factor):
        for x in range(0, mask.shape[1], factor):
            mask_window = mask[y:y+factor,x:x+factor]
            roi_window = roi_morphed[y:y+factor,x:x+factor]
            
            if mask_window.mean() > relevance_cutoff:
                mask_downsampled[int(y/factor),int(x/factor)] = 1

            elif mask_window.mean() <= relevance_cutoff:
                pass
            
            if roi_window.mean() > relevance_cutoff:
                roi_downsampled[int(y/factor),int(x/factor)] = 1

            elif roi_window.mean() <= relevance_cutoff:
                pass            
            
    #Second downsampling
    mask_downsampled2x = np.zeros(shape=(int((mask_downsampled.shape[0]-1)/factor), int(mask_downsampled.shape[1]/factor)))
    roi_downsampled2x = np.zeros(shape=(int((roi_downsampled.shape[0]-1)/factor), int(roi_downsampled.shape[1]/factor)))
    for y in range(0, (mask_downsampled.shape[0] - 1), factor):
        for x in range(0, mask_downsampled.shape[1], factor):
            mask_window = mask_downsampled[y:y+factor,x:x+factor]
            roi_window = roi_downsampled[y:y+factor,x:x+factor]
            if mask_window.mean() > relevance_cutoff:
                mask_downsampled2x[int(y/factor),int(x/factor)] = 1

            elif mask_window.mean() <= relevance_cutoff:
                pass  
            
            if roi_window.mean() > relevance_cutoff:
                roi_downsampled2x[int(y/factor),int(x/factor)] = 1

            elif roi_window.mean() <= relevance_cutoff:
                pass 
            
    return mask, mask_downsampled, mask_downsampled2x, roi_morphed, roi_downsampled, roi_downsampled2x

def generate_masks_2(relevance_cutoff):
    """Generates masks for widefield images of original size, downsampled, and downsampled twice.
    relevance_cutoff determines which pixels are considered as masked when downsampling.
    Only difference stems from alternative ROI being applied
    """
    roi_morphed = np.load(open("auditorycoding/F189/masks/1_roi_morphed_new.npy", "rb"))
    mask_stabilized = np.load(open("auditorycoding/F189/masks/mask_stabilized.npy", "rb"))   
    mask = roi_morphed * mask_stabilized[:,:,1]
    factor = 2
    roi_downsampled = np.zeros(shape=(int(mask.shape[0]/factor), int(mask.shape[1]/factor)))
    mask_downsampled = np.zeros(shape=(int(mask.shape[0]/factor), int(mask.shape[1]/factor)))
    for y in range(0, mask.shape[0], factor):
        for x in range(0, mask.shape[1], factor):
            mask_window = mask[y:y+factor,x:x+factor]
            roi_window = roi_morphed[y:y+factor,x:x+factor]
            
            if mask_window.mean() > relevance_cutoff:
                mask_downsampled[int(y/factor),int(x/factor)] = 1

            elif mask_window.mean() <= relevance_cutoff:
                pass
            
            if roi_window.mean() > relevance_cutoff:
                roi_downsampled[int(y/factor),int(x/factor)] = 1

            elif roi_window.mean() <= relevance_cutoff:
                pass            
            
    #Second downsampling
    mask_downsampled2x = np.zeros(shape=(int((mask_downsampled.shape[0]-1)/factor), int(mask_downsampled.shape[1]/factor)))
    roi_downsampled2x = np.zeros(shape=(int((roi_downsampled.shape[0]-1)/factor), int(roi_downsampled.shape[1]/factor)))
    for y in range(0, (mask_downsampled.shape[0] - 1), factor):
        for x in range(0, mask_downsampled.shape[1], factor):
            mask_window = mask_downsampled[y:y+factor,x:x+factor]
            roi_window = roi_downsampled[y:y+factor,x:x+factor]
            if mask_window.mean() > relevance_cutoff:
                mask_downsampled2x[int(y/factor),int(x/factor)] = 1

            elif mask_window.mean() <= relevance_cutoff:
                pass  
            
            if roi_window.mean() > relevance_cutoff:
                roi_downsampled2x[int(y/factor),int(x/factor)] = 1

            elif roi_window.mean() <= relevance_cutoff:
                pass 
            
    return mask, mask_downsampled, mask_downsampled2x, roi_morphed, roi_downsampled, roi_downsampled2x

def preprocessing(tseries, spectrograms, relevance_cutoff = 0.25, delay=0, ds=4, input_seconds=0.3, mode="filtered",
                all_pure=False):
    """Generates the pair of datasets for a given tseries (X and Y). All sounds are treated as spectrograms.
    
    tseries = name of the tsearie 
    spectrograms = dictionary created by create_spectrograms
    relevance_cutoff = masked are during downsampling
    delay = delay between widefield image and stimulus in sec
    ds = downsampling factor
    input_seconds = length of spectrogram before widefield image
    mode = ("raw", "filtered", "deconv")
    all_pure = indicator for pure data only
    
    Based on the datatype (complex or pure) used we create arrays indicating which stimulus and
    trial belong to each frame. For each time frame we create dictionary entries indicating stimulus ID and trial.
    """
    if int(tseries[-2:]) in [21, 23, 24, 26, 27, 29, 31, 32, 35, 36, 37, 38]:
        # These are complex stimuli with corresponding audio files
        frametimes = np.load("auditorycoding/F189/"+tseries+"/frametimes.npy")
        stimulus = np.load("auditorycoding/F189/"+tseries+"/stimulus.npy")
        stimparams = pickle.load(open("auditorycoding/F189/"+tseries+"/stimparams.dict", "rb"))  
        numTrials  = stimparams['numTrials']
        numAudioStims = stimparams['numAudioStims']
        trials = np.array([int(i/numAudioStims) for i in range(numTrials * numAudioStims)]) +1 

        trial_index = {}
        stimulus_timed = {}
        stimulus_index = {}
        for idx, stim in enumerate(stimulus):
            start = stim[1]
            endtimes = spectrograms[stim[0]][3]
            synced_times = endtimes + start
            for j in range(len(synced_times)):
                stimulus_timed[synced_times[j]] = spectrograms[stim[0]][2][:,j]
                stimulus_index[synced_times[j]] = stim[0]
                trial_index[synced_times[j]] = trials[idx]

    elif int(tseries[-2:]) in [28, 33, 34, 39]:
        # These are pure tone stimuli where no input file exists either get 
        # pure tone .wav files or somehow generate a spectrogram with amp
        # being the average peak-amp of the complex files
        frametimes = np.load("auditorycoding/F189/"+tseries+"/frametimes.npy")
        stimulus = np.load("auditorycoding/F189/"+tseries+"/stimulus.npy")
        stimparams = pickle.load(open("auditorycoding/F189/"+tseries+"/stimparams.dict", "rb"))
        numTrials  = stimparams['numTrials']
        numAudioStims = stimparams['numFrequencies']
        trials = np.array([int(i/numAudioStims) for i in range(numTrials * numAudioStims)]) +1 

        path = "auditorycoding/audiostimfiles/ComplexAudioStimFiles_03_2021/"
        stimulus_names = [file for file in os.listdir(path)
                        if os.path.isfile(os.path.join(path, file))]
        complex_idx = len(stimulus_names) - 1

        trial_index = {}
        stimulus_timed = {}
        stimulus_index = {}
        for i, stim in enumerate(stimulus):
            idx = stim[0] + complex_idx
            start = stim[1]
            endtimes = spectrograms[idx][3]
            synced_times = endtimes + start
            for j in range(len(synced_times)):
                stimulus_timed[synced_times[j]] = spectrograms[idx][2][:,j]
                stimulus_index[synced_times[j]] = idx
                trial_index[synced_times[j]] = trials[i]

    elif int(tseries[-2:]) in [25, 30]:
        # These are based on video files, usure how to handle
        frametimes = np.load("auditorycoding/F189/"+tseries+"/frametimes.npy")
        stimulus = np.load("auditorycoding/F189/"+tseries+"/stimulus.npy")
        stimparams = pickle.load(open("auditorycoding/F189/"+tseries+"/stimparams.dict", "rb"))

        stimulus_timed = {}
        stimulus_index = {}
        for i in stimulus:
            start = i[1]
            endtimes = spectrograms[i[0]][3]
            synced_times = endtimes + start
            for j in range(len(synced_times)):
                stimulus_timed[synced_times[j]] = spectrograms[i[0]][2][:,j]
                stimulus_index[synced_times[j]] = i[0]
                
    # Load respective files
    # Depending on what images we use (downsampling, filtering) we load in the tseries, flatten it,
    # and remove the masked pixels.

    mask, mask_downsampled, mask_downsampled2x, roi, roid, roid2x = generate_masks(relevance_cutoff)

    if int(tseries[-2:]) in [25, 28, 33]:
        DF_by_F0 = "DF_by_F0_201_50"

    elif int(tseries[-2:]) not in [25, 28, 33]:
        DF_by_F0 = "DF_by_F0_351_50"

    if mode == "raw":
        mask, mask_downsampled, mask_downsampled2x, roi, roid, roid2x = generate_masks(relevance_cutoff)
        if ds == 2:
            DF_by_F0_351_50_downsampled = pickle.load(open(f"auditorycoding/F189/{tseries}/{DF_by_F0}_downsampled.p", "rb"))
            DF_by_F0_351_50_downsampled = np.reshape(DF_by_F0_351_50_downsampled, 
                                            (DF_by_F0_351_50_downsampled.shape[0],
                                            DF_by_F0_351_50_downsampled.shape[1]* DF_by_F0_351_50_downsampled.shape[2]))
            mask_downsampled = np.reshape(mask_downsampled, mask_downsampled.shape[0]* mask_downsampled.shape[1])
            mask_downsampled = np.argwhere(mask_downsampled == 0).flatten()
            DF_by_F0_351_50_downsampled = np.delete(DF_by_F0_351_50_downsampled, mask_downsampled, axis=1)
            DF_by_F0_351_50_downsampled = np.nan_to_num(DF_by_F0_351_50_downsampled, nan=np.nanmean(DF_by_F0_351_50_downsampled))
            print("Shape of Y, downsampled, mask cut out: ", DF_by_F0_351_50_downsampled.shape)
                    
        elif ds == 4:
            DF_by_F0_351_50_downsampled2x = pickle.load(open(f"auditorycoding/F189/{tseries}/{DF_by_F0}_downsampled2x_nomask.p", "rb"))
            DF_by_F0_351_50_downsampled2x = np.reshape(DF_by_F0_351_50_downsampled2x, 
                                            (DF_by_F0_351_50_downsampled2x.shape[0],
                                            DF_by_F0_351_50_downsampled2x.shape[1]* DF_by_F0_351_50_downsampled2x.shape[2]))  
            mask_downsampled2x = np.reshape(mask_downsampled2x, mask_downsampled2x.shape[0]* mask_downsampled2x.shape[1])
            mask_downsampled2x = np.argwhere(mask_downsampled2x == 0).flatten()
            DF_by_F0_351_50_downsampled2x = np.delete(DF_by_F0_351_50_downsampled2x, mask_downsampled2x, axis=1)
            DF_by_F0_351_50_downsampled2x = np.nan_to_num(DF_by_F0_351_50_downsampled2x, nan=np.nanmean(DF_by_F0_351_50_downsampled2x))
            print("Shape of Y, downsampled2x, mask cut out: ", DF_by_F0_351_50_downsampled2x.shape)
        # Remove nan values from the downsampled images and cut out mask
    
    if mode == "deconv":
        mask, mask_downsampled, mask_downsampled2x, roi, roid, roid2x = generate_masks_2(relevance_cutoff)
        if ds == 2:
            print("Not Implemented Yet")
            
        if ds == 4:
            DF_by_F0_351_50_downsampled2x = pickle.load(open(f"auditorycoding/F189/{tseries}/firdif_189_{tseries[-2:]}_norm_roi_downsampled2x_filtered_0.25.p", "rb"))
            DF_by_F0_351_50_downsampled2x = np.reshape(DF_by_F0_351_50_downsampled2x, 
                                            (DF_by_F0_351_50_downsampled2x.shape[0],
                                            DF_by_F0_351_50_downsampled2x.shape[1]* DF_by_F0_351_50_downsampled2x.shape[2]))  
            mask_downsampled2x = np.reshape(mask_downsampled2x, mask_downsampled2x.shape[0]* mask_downsampled2x.shape[1])
            mask_downsampled2x = np.argwhere(mask_downsampled2x == 0).flatten()
            
            DF_by_F0_351_50_downsampled2x = np.delete(DF_by_F0_351_50_downsampled2x, mask_downsampled2x, axis=1)
            DF_by_F0_351_50_downsampled2x = np.nan_to_num(DF_by_F0_351_50_downsampled2x, nan=np.nanmean(DF_by_F0_351_50_downsampled2x))
            
    if mode == "filtered":
        mask, mask_downsampled, mask_downsampled2x, roi, roid, roid2x = generate_masks(relevance_cutoff)
        if ds == 2:
            print("Not Implemented Yet")
            
        if ds == 4:
            DF_by_F0_351_50_downsampled2x = pickle.load(open(f"auditorycoding/F189/{tseries}/{DF_by_F0}_downsampled2x_filtered_{relevance_cutoff}.p", "rb"))
            DF_by_F0_351_50_downsampled2x = np.reshape(DF_by_F0_351_50_downsampled2x, 
                                            (DF_by_F0_351_50_downsampled2x.shape[0],
                                            DF_by_F0_351_50_downsampled2x.shape[1]* DF_by_F0_351_50_downsampled2x.shape[2]))  
            mask_downsampled2x = np.reshape(mask_downsampled2x, mask_downsampled2x.shape[0]* mask_downsampled2x.shape[1])
            mask_downsampled2x = np.argwhere(mask_downsampled2x == 0).flatten()
            
            DF_by_F0_351_50_downsampled2x = np.delete(DF_by_F0_351_50_downsampled2x, mask_downsampled2x, axis=1)
            DF_by_F0_351_50_downsampled2x = np.nan_to_num(DF_by_F0_351_50_downsampled2x, nan=np.nanmean(DF_by_F0_351_50_downsampled2x))
    

    # Create Trial index for each stimulus

    # all time stamps of the stimuli
    keys = np.array([key for key in stimulus_timed])
    # Create the final input matrix by matching sounds (with delay) and image frames

    input_len = int(15 * input_seconds) #input duration in frames
    X = np.zeros([len(frametimes), len(stimulus_timed[keys[0]]) * input_len])
    frame_index = np.zeros(len(frametimes)) # stimulus index per frame
    frame_trial_index = np.zeros(len(frametimes)) # trial index per frame

    pauses = [] 
    # for each frame we inspect the time (input_secodns) before and find time frames
    # from spectrograms before them. If there are frames missing we append the index
    # of the frame to pauses which alre deleted. This generates input data X.
    for i in range(len(frametimes)):
        try:
            closest_index = np.argmax(keys[keys < (frametimes[i] - delay)])
            furthest_index = max(0 , closest_index - (input_len - 1))
            closest_smaller_soundframe = keys[closest_index]
            furthest_smaller_soundframe = keys[furthest_index]
            gap = 0 # indicates the gap b/w the frame and closest smaller sound (seconds)
            gap_index = 0 # indicates the gap b/w the frame and closest smaller sound (#Frames)
            while True:
                if (frametimes[i] - delay) - closest_smaller_soundframe - gap <= 1/15: 
                    break
                elif (frametimes[i] - delay) - closest_smaller_soundframe - gap > 1/15:
                    gap_index += 1
                    gap += 1/15

                if gap_index == input_len:
                    break

            spectrogram_keys = keys[furthest_index+gap_index:closest_index+1]
        
            if len(spectrogram_keys) < input_len and gap == 0:
                matched_spectrogram = np.vstack((np.zeros((input_len-len(spectrogram_keys),len(spectrograms[0][1]))),
                                                     np.array([stimulus_timed[s] for s in spectrogram_keys]))).T

                pauses.append(i)
            else:
                front_index = 0

                for k in range(len(spectrogram_keys)-1, 0, -1):
                    if (spectrogram_keys[k] - spectrogram_keys[k-1]) > (1/15+0.1):
                        spectrogram_keys = spectrogram_keys[k:]
                        front_index = k
                        break

                if gap_index == 0:
                    if front_index == 0:  
                        matched_spectrogram = np.array([stimulus_timed[s] for s in spectrogram_keys]).T

                    elif front_index != 0:
                        pauses.append(i)
                        matched_spectrogram = np.vstack((np.zeros((front_index,len(spectrograms[0][1]))),
                                                         np.array([stimulus_timed[s] for s in spectrogram_keys]))).T
                        
                elif gap_index > 0 and gap_index < input_len:
                    pauses.append(i)
                    if front_index == 0:
                        matched_spectrogram = np.vstack((np.array([stimulus_timed[s] for s in spectrogram_keys]), 
                                                         np.zeros((gap_index, len(spectrograms[0][1]))))).T 
                        
                    elif front_index != 0:
                        matched_spectrogram = np.vstack((np.zeros((front_index, len(spectrograms[0][1]))),
                                                        np.array([stimulus_timed[s] for s in spectrogram_keys]), 
                                                         np.zeros((gap_index, len(spectrograms[0][1]))))).T  
                elif gap_index == input_len:
                    matched_spectrogram = np.zeros((gap_index,len(spectrograms[0][1])))
                    pauses.append(i)

            if all_pure == False:
                if int(tseries[-2:]) in [28, 33, 34, 39]:
                    # Exclude Frames with high frequencies
                    if matched_spectrogram[-1,-1] != 0:
                        pauses.append(i)
                    


            X[i,:] = matched_spectrogram.flatten()
            frame_index_list = [stimulus_index[j] for j in spectrogram_keys]
            frame_index[i] = stats.mode(frame_index_list)[0]

            trial_index_list = [trial_index[j] for j in spectrogram_keys]
            frame_trial_index[i] = stats.mode(trial_index_list)[0]
            if any([1 for j in frame_index_list if j != stats.mode(frame_index_list)[0]]):
                print(i,frame_index_list)
            
        except ValueError:
            #print("ValueError: ", frametimes[i])
            pauses.append(i)
            
    # delete pauses from all frames        
    X_cut = np.delete(X, pauses, axis=0)
    frametimes_cut = np.delete(frametimes, pauses)

    frame_index_cut = np.delete(frame_index, pauses)
    frame_trial_index_cut = np.delete(frame_trial_index, pauses)

    if ds == 2:
        print("Shape of Y: ", DF_by_F0_351_50_downsampled.shape)
    
        return X, DF_by_F0_351_50_downsampled

    elif ds == 4:
        DF_by_F0_351_50_downsampled2x_cut = np.delete(DF_by_F0_351_50_downsampled2x, pauses, axis=0)
        return X, DF_by_F0_351_50_downsampled2x, X_cut, DF_by_F0_351_50_downsampled2x_cut, frametimes_cut, frame_index_cut, frame_trial_index_cut
    


def preprocessing_pure(tseries, relevance_cutoff = 0.5, delay=0, ds=4, mode="raw"):
    """ Identical to above function with few exceptions. Only pure tones can be applied
    and are not considered as spectrograms but input matrix X only contains one column
    with the frequency of the stimulus currently present.
    """

    # These are pure tone stimuli where no input file exists either get 
    # pure tone .wav files or somehow generate a spectrogram with amp
    # being the average peak-amp of the complex files
    frametimes = np.load("auditorycoding/F189/"+tseries+"/frametimes.npy")
    stimulus = np.load("auditorycoding/F189/"+tseries+"/stimulus.npy")
    stimparams = pickle.load(open("auditorycoding/F189/"+tseries+"/stimparams.dict", "rb"))
    numTrials  = stimparams['numTrials']
    numAudioStims = stimparams['numFrequencies']
    trials = np.array([int(i/numAudioStims) for i in range(numTrials * numAudioStims)]) +1 

    # Load respective files

    mask, mask_downsampled, mask_downsampled2x, roi, roid, roid2x = generate_masks(relevance_cutoff)

    if int(tseries[-2:]) in [25, 28, 33]:
        DF_by_F0 = "DF_by_F0_201_50"

    elif int(tseries[-2:]) not in [25, 28, 33]:
        DF_by_F0 = "DF_by_F0_351_50"

    if mode == "raw":
        if ds == 2:
            DF_by_F0_351_50_downsampled = pickle.load(open(f"auditorycoding/F189/{tseries}/{DF_by_F0}_downsampled.p", "rb"))
            DF_by_F0_351_50_downsampled = np.reshape(DF_by_F0_351_50_downsampled, 
                                            (DF_by_F0_351_50_downsampled.shape[0],
                                            DF_by_F0_351_50_downsampled.shape[1]* DF_by_F0_351_50_downsampled.shape[2]))
            mask_downsampled = np.reshape(mask_downsampled, mask_downsampled.shape[0]* mask_downsampled.shape[1])
            mask_downsampled = np.argwhere(mask_downsampled == 0).flatten()
            DF_by_F0_351_50_downsampled = np.delete(DF_by_F0_351_50_downsampled, mask_downsampled, axis=1)
            DF_by_F0_351_50_downsampled = np.nan_to_num(DF_by_F0_351_50_downsampled, nan=np.nanmean(DF_by_F0_351_50_downsampled))
            print("Shape of Y, downsampled, mask cut out: ", DF_by_F0_351_50_downsampled.shape)
                    
        elif ds == 4:
            DF_by_F0_351_50_downsampled2x = pickle.load(open(f"auditorycoding/F189/{tseries}/{DF_by_F0}_downsampled2x_nomask.p", "rb"))
            DF_by_F0_351_50_downsampled2x = np.reshape(DF_by_F0_351_50_downsampled2x, 
                                            (DF_by_F0_351_50_downsampled2x.shape[0],
                                            DF_by_F0_351_50_downsampled2x.shape[1]* DF_by_F0_351_50_downsampled2x.shape[2]))  
            mask_downsampled2x = np.reshape(mask_downsampled2x, mask_downsampled2x.shape[0]* mask_downsampled2x.shape[1])
            mask_downsampled2x = np.argwhere(mask_downsampled2x == 0).flatten()
            DF_by_F0_351_50_downsampled2x = np.delete(DF_by_F0_351_50_downsampled2x, mask_downsampled2x, axis=1)
            DF_by_F0_351_50_downsampled2x = np.nan_to_num(DF_by_F0_351_50_downsampled2x, nan=np.nanmean(DF_by_F0_351_50_downsampled2x))
            print("Shape of Y, downsampled2x, mask cut out: ", DF_by_F0_351_50_downsampled2x.shape)
        # Remove nan values from the downsampled images and cut out mask
    
    if mode == "filtered":
        if ds == 2:
            print("Not Implemented Yet")
            
        if ds == 4:
            DF_by_F0_351_50_downsampled2x = pickle.load(open(f"auditorycoding/F189/{tseries}/{DF_by_F0}_downsampled2x_filtered_{relevance_cutoff}.p", "rb"))
            DF_by_F0_351_50_downsampled2x = np.reshape(DF_by_F0_351_50_downsampled2x, 
                                            (DF_by_F0_351_50_downsampled2x.shape[0],
                                            DF_by_F0_351_50_downsampled2x.shape[1]* DF_by_F0_351_50_downsampled2x.shape[2]))  
            mask_downsampled2x = np.reshape(mask_downsampled2x, mask_downsampled2x.shape[0]* mask_downsampled2x.shape[1])
            mask_downsampled2x = np.argwhere(mask_downsampled2x == 0).flatten()
            
            DF_by_F0_351_50_downsampled2x = np.delete(DF_by_F0_351_50_downsampled2x, mask_downsampled2x, axis=1)
            DF_by_F0_351_50_downsampled2x = np.nan_to_num(DF_by_F0_351_50_downsampled2x, nan=np.nanmean(DF_by_F0_351_50_downsampled2x))

    
    # Create the final input matrix by matching sounds (with delay) and image frames

    X = np.zeros([len(frametimes), 1])
    frame_index = np.zeros(len(frametimes))
    frame_trial_index = np.zeros(len(frametimes))

    pure_frequencies = {1: 100.0, 2: 156.71210662, 3: 245.58684361, 4: 384.86431619, 5: 603.12897753, 
                    6: 945.17612631, 7: 1481.20541879, 8: 2321.22821513, 9: 3637.64563534,
                    10: 5700.63110645,11: 8933.57909744,12: 14000.}

         
    for i in range(len(stimulus)):
        stim = stimulus[i][0]
        trial = trials[i]
        t = stimulus[i][1]
        indices = np.where(np.logical_and(frametimes>=t, frametimes<=(t+1)))
        # if delay is wanted in this scenario
        #indices = np.where(np.logical_and((frametimes-delay)>=t, (frametimes-delay)<=(t+1)))
        X[indices,:] = pure_frequencies[stim]
        frame_index[indices] = stim
        frame_trial_index[indices] = trial
        

    pauses = np.where(frame_index == 0)

            
    X_cut = np.delete(X, pauses, axis=0)
    frametimes_cut = np.delete(frametimes, pauses)
    frame_index_cut = np.delete(frame_index, pauses)
    frame_trial_index_cut = np.delete(frame_trial_index, pauses)

    if ds == 2:
        #DF_by_F0_351_50_downsampled_cut = np.delete(DF_by_F0_351_50_downsampled, pauses, axis=0)
        print("Shape of Y: ", DF_by_F0_351_50_downsampled.shape)
    
        return X, DF_by_F0_351_50_downsampled

    elif ds == 4:
        DF_by_F0_351_50_downsampled2x_cut = np.delete(DF_by_F0_351_50_downsampled2x, pauses, axis=0)
        return X, DF_by_F0_351_50_downsampled2x, X_cut, DF_by_F0_351_50_downsampled2x_cut, frametimes_cut, frame_index_cut, frame_trial_index_cut

def preprocessing_mps(tseries, relevance_cutoff = 0.25, delay=0.1, ds=4, mode="filtered"):
    """ Identical to above function with few exceptions. Only complex tones can be applied
    and are not considered as spectrograms but input matrix X contains the modulation 
    power spectroum of the stimulus currently present. Delay has no function.
    """

    # These are pure tone stimuli where no input file exists either get 
    # pure tone .wav files or somehow generate a spectrogram with amp
    # being the average peak-amp of the complex files
    frametimes = np.load("auditorycoding/F189/"+tseries+"/frametimes.npy")
    stimulus = np.load("auditorycoding/F189/"+tseries+"/stimulus.npy")
    stimparams = pickle.load(open("auditorycoding/F189/"+tseries+"/stimparams.dict", "rb"))
    numTrials  = stimparams['numTrials']
    numAudioStims = stimparams['numAudioStims']
    trials = np.array([int(i/numAudioStims) for i in range(numTrials * numAudioStims)]) +1 

    # Load respective files

    mask, mask_downsampled, mask_downsampled2x, roi, roid, roid2x = generate_masks(relevance_cutoff)

    if int(tseries[-2:]) in [25, 28, 33]:
        DF_by_F0 = "DF_by_F0_201_50"

    elif int(tseries[-2:]) not in [25, 28, 33]:
        DF_by_F0 = "DF_by_F0_351_50"

    if mode == "raw":
        if ds == 2:
            DF_by_F0_351_50_downsampled = pickle.load(open(f"auditorycoding/F189/{tseries}/{DF_by_F0}_downsampled.p", "rb"))
            DF_by_F0_351_50_downsampled = np.reshape(DF_by_F0_351_50_downsampled, 
                                            (DF_by_F0_351_50_downsampled.shape[0],
                                            DF_by_F0_351_50_downsampled.shape[1]* DF_by_F0_351_50_downsampled.shape[2]))
            mask_downsampled = np.reshape(mask_downsampled, mask_downsampled.shape[0]* mask_downsampled.shape[1])
            mask_downsampled = np.argwhere(mask_downsampled == 0).flatten()
            DF_by_F0_351_50_downsampled = np.delete(DF_by_F0_351_50_downsampled, mask_downsampled, axis=1)
            DF_by_F0_351_50_downsampled = np.nan_to_num(DF_by_F0_351_50_downsampled, nan=np.nanmean(DF_by_F0_351_50_downsampled))
            print("Shape of Y, downsampled, mask cut out: ", DF_by_F0_351_50_downsampled.shape)
                    
        elif ds == 4:
            DF_by_F0_351_50_downsampled2x = pickle.load(open(f"auditorycoding/F189/{tseries}/{DF_by_F0}_downsampled2x_nomask.p", "rb"))
            DF_by_F0_351_50_downsampled2x = np.reshape(DF_by_F0_351_50_downsampled2x, 
                                            (DF_by_F0_351_50_downsampled2x.shape[0],
                                            DF_by_F0_351_50_downsampled2x.shape[1]* DF_by_F0_351_50_downsampled2x.shape[2]))  
            mask_downsampled2x = np.reshape(mask_downsampled2x, mask_downsampled2x.shape[0]* mask_downsampled2x.shape[1])
            mask_downsampled2x = np.argwhere(mask_downsampled2x == 0).flatten()
            DF_by_F0_351_50_downsampled2x = np.delete(DF_by_F0_351_50_downsampled2x, mask_downsampled2x, axis=1)
            DF_by_F0_351_50_downsampled2x = np.nan_to_num(DF_by_F0_351_50_downsampled2x, nan=np.nanmean(DF_by_F0_351_50_downsampled2x))
            print("Shape of Y, downsampled2x, mask cut out: ", DF_by_F0_351_50_downsampled2x.shape)
        # Remove nan values from the downsampled images and cut out mask
    
    if mode == "filtered":
        if ds == 2:
            print("Not Implemented Yet")
            
        if ds == 4:
            DF_by_F0_351_50_downsampled2x = pickle.load(open(f"auditorycoding/F189/{tseries}/{DF_by_F0}_downsampled2x_filtered_{relevance_cutoff}.p", "rb"))
            DF_by_F0_351_50_downsampled2x = np.reshape(DF_by_F0_351_50_downsampled2x, 
                                            (DF_by_F0_351_50_downsampled2x.shape[0],
                                            DF_by_F0_351_50_downsampled2x.shape[1]* DF_by_F0_351_50_downsampled2x.shape[2]))  
            mask_downsampled2x = np.reshape(mask_downsampled2x, mask_downsampled2x.shape[0]* mask_downsampled2x.shape[1])
            mask_downsampled2x = np.argwhere(mask_downsampled2x == 0).flatten()
            
            DF_by_F0_351_50_downsampled2x = np.delete(DF_by_F0_351_50_downsampled2x, mask_downsampled2x, axis=1)
            DF_by_F0_351_50_downsampled2x = np.nan_to_num(DF_by_F0_351_50_downsampled2x, nan=np.nanmean(DF_by_F0_351_50_downsampled2x))
    
    # Create the final input matrix by matching sounds (with delay) and image frames

    X = np.zeros([len(frametimes), 77*101])
    frame_index = np.zeros(len(frametimes))
    frame_trial_index = np.zeros(len(frametimes))


    pure_frequencies = {1: 100.0, 2: 156.71210662, 3: 245.58684361, 4: 384.86431619, 5: 603.12897753, 
                    6: 945.17612631, 7: 1481.20541879, 8: 2321.22821513, 9: 3637.64563534,
                    10: 5700.63110645,11: 8933.57909744,12: 14000.}
    mps_dict = pickle.load(open("mps_dict.p", "rb"))
    path = "auditorycoding/audiostimfiles/ComplexAudioStimFiles_03_2021/"
    stimulus_names = [file for file in os.listdir(path)
                     if os.path.isfile(os.path.join(path, file))]
    duration_dict = {}
    for i in stimparams["stimDuration"]:
        duration_dict[i[0]] = i[1]
    for i in range(len(stimulus)):
        stim = stimulus[i][0]
        trial = trials[i]
        t = stimulus[i][1]
        name = stimulus_names[stim][:-4]
        duration = duration_dict[stim]
        indices = np.where(np.logical_and(frametimes>=t, frametimes<=(t+duration)))
        X[indices,:] = mps_dict[name].flatten()
        frame_index[indices] = stim
        frame_trial_index[indices] = trial
        

    pauses = np.where(frame_index == 0)

            
    X_cut = np.delete(X, pauses, axis=0)

    frametimes_cut = np.delete(frametimes, pauses)
    frame_index_cut = np.delete(frame_index, pauses)
    frame_trial_index_cut = np.delete(frame_trial_index, pauses)

    if ds == 2:
        #DF_by_F0_351_50_downsampled_cut = np.delete(DF_by_F0_351_50_downsampled, pauses, axis=0)
        print("Shape of Y: ", DF_by_F0_351_50_downsampled.shape)
    
        return X, DF_by_F0_351_50_downsampled

    elif ds == 4:
        DF_by_F0_351_50_downsampled2x_cut = np.delete(DF_by_F0_351_50_downsampled2x, pauses, axis=0)
        return X, DF_by_F0_351_50_downsampled2x, X_cut, DF_by_F0_351_50_downsampled2x_cut, frametimes_cut, frame_index_cut, frame_trial_index_cut


def reverse_preprocessing_img(Y, relevance_cutoff = 0.25, ds=4):
    """ Reverses the preprocessing of a two dimensional array Y of widefield images.
    It turns it back into a 3d array with 2d images in the last 2 dimensions, inserts frames
    cut due to masking.
    """
    mask, mask_downsampled, mask_downsampled2x, roi, roid, roid2x = generate_masks(relevance_cutoff)
            
            
    if ds == 2:
        target_shape = mask_downsampled.shape
        mask_downsampled = np.reshape(mask_downsampled, mask_downsampled.shape[0]* mask_downsampled.shape[1])
        #mask_downsampled = np.argwhere(mask_downsampled == 0).flatten()

        for i in tqdm(range(mask_downsampled.shape[0])):
            if mask_downsampled[i] == 0:
                Y = np.insert(Y, i, np.nan, axis=1)
        print(Y.shape, mask_downsampled.shape)
        Y = np.reshape(Y,(Y.shape[0], int(target_shape[0]), int(target_shape[1])))        
        return Y
    
    if ds == 4:
        target_shape = mask_downsampled2x.shape
        mask_downsampled2x = np.reshape(mask_downsampled2x, mask_downsampled2x.shape[0]* mask_downsampled2x.shape[1])
        for i in tqdm(range(mask_downsampled2x.shape[0])):
            if mask_downsampled2x[i] == 0:
                Y = np.insert(Y, i, np.nan, axis=1)
        print(Y.shape, mask_downsampled2x.shape)
        Y = np.reshape(Y,(Y.shape[0], int(target_shape[0]), int(target_shape[1])))        
        return Y

def reverse_cut(X, pauses):
    """ Inserts empty colums into arrays where pauses have been cut
    to recreate original shape.
    """
    for p in tqdm(pauses):
        X = np.insert(X,p,0, axis=0)
        
    return X


def cv_split(X, Y):
    """ Splits data in to train and test sets using random 5-fold cross-validation.
    """
    splits_X = []
    splits_Y = []
    indices = np.random.permutation(X.shape[0])
    r = np.linspace(0+1/5,1,5)
    length = X.shape[0]
    for i in range(len(r)):
        if i == 0:
            test_idx = indices[0:int(r[i]*length)]
            training_idx = indices[int(r[i]*length):]
        else:
            test_idx = indices[int(r[i-1]*length):int(r[i]*length)]
            training_idx = np.append(indices[0:int(r[i-1]*length)], indices[int(r[i]*length):])
        
        X_training, X_test = X[training_idx,:], X[test_idx,:]
        Y_training, Y_test = Y[training_idx,:], Y[test_idx,:]
        splits_X.append((X_training, X_test))
        splits_Y.append((Y_training, Y_test))
    return splits_X, splits_Y


def cv_split_trial(X, Y, frame_trial_index_cut, index_set=None, overall_idx=False, mode = "drive"):
    """ Splits data based on trials from frame_trial_index_cut array. 
    mode determines whether all sets are kept in RAM or instead saved on disk.
    index_set is the set of trial IDs, leave as None.
    splits overall tracks the index of each frame as originally given.
    
    """
    splits_X = []
    splits_Y = []
    if type(overall_idx) != bool:
        splits_overall = []

    if index_set == None:
        index_set = set(frame_trial_index_cut)
    for i in index_set:
        train_idx = np.where(frame_trial_index_cut != i)[0]
        test_idx = np.where(frame_trial_index_cut == i)[0]
        X_training, X_test = X[train_idx], X[test_idx]
        Y_training, Y_test = Y[train_idx], Y[test_idx]
        if type(overall_idx) != bool:
            splits_overall.append((overall_idx[train_idx], overall_idx[test_idx]))

        if mode == "drive":
            np.save(f"temp/X_train_{i}.npy", X_training)
            np.save(f"temp/X_test_{i}.npy", X_test)
            np.save(f"temp/Y_train_{i}.npy", Y_training)
            np.save(f"temp/Y_test_{i}.npy", Y_test)
            #splits_X.append((X_training, X_test))
            #splits_Y.append((Y_training, Y_test))
            splits_X.append((f"temp/X_train_{i}.npy", f"temp/X_test_{i}.npy"))
            splits_Y.append((f"temp/Y_train_{i}.npy", f"temp/Y_test_{i}.npy"))

        elif mode == "ram":
            splits_X.append((X_training, X_test))
            splits_Y.append((Y_training, Y_test))
        
    if type(overall_idx) == bool:
        return splits_X, splits_Y

    if type(overall_idx) != bool:
        return splits_X, splits_Y, splits_overall

def cv_split_newstim(X, Y, frame_index_cut,
                     frame_trial_index_cut,
                     index_set=None,
                     mode = "drive",
                     reverse_direction=False,
                     give_indices=False):
    """Cross-validation splits are returned based on stimuli. Test sets contain 4 stimuli each
    train sets the raimaining 40.
    reverse_direction indicates backward predictions
    This split method may only be used for the complex tones, because duplicate frames are
    eliminated in the test sets. Pure tones only cosist of duplicates."""
    trials = set(frame_trial_index_cut)
    n_trials = len(set(frame_trial_index_cut))

    splits_X = []
    splits_Y = []

    if index_set == None:
        index_set = set(frame_index_cut)
        index_list = np.array(list(index_set))

    test_sets = np.array(np.split(index_list, len(index_set) / 4))
    train_sets = np.array([list(index_set.difference(i)) for i in test_sets])

    idx_per_set = []
    for idx in range(len(test_sets)):
        idx_subset = []
        train_idx = [j for j in range(len(frame_index_cut)) if any(frame_index_cut[j] == i for i in train_sets[idx])]
        X_training = X[train_idx]
        Y_training = Y[train_idx]

        X_test = False
        Y_test = False
        for stim_idx, stim in enumerate(test_sets[idx]):
            test_idx = np.where(frame_index_cut == stim)[0]
            
            test_trial_index = frame_trial_index_cut[test_idx]

            x_test = X[test_idx]
            y_test = Y[test_idx]
            len_per_trial = [len(np.where(test_trial_index == k)[0]) for k in set(test_trial_index)]

            reference_trial = list(trials)[0]
            reference_condition = np.where(test_trial_index == reference_trial)[0]
            idx_subset.extend(np.ones(len(reference_condition))*stim)
            if reverse_direction == False:
                x_len = np.unique(x_test[reference_condition,:], axis=0)
                y_len = y_test[reference_condition,:]

                x_new = np.zeros(shape=(n_trials, x_len.shape[0], x_len.shape[1]))
                y_new = np.zeros(shape=(n_trials, x_len.shape[0], y_len.shape[1]))

                for t, trial_number in enumerate(trials):
                    x_temp = x_test[np.where(test_trial_index == trial_number)[0],:]
                    y_temp = y_test[np.where(test_trial_index == trial_number)[0],:]

                    _, unique_idx = np.unique(x_temp, return_index=True, axis=0)
                    uniquey = np.unique(y_temp, axis=0)
                    x_new[int(t),:,:] = x_temp[np.sort(unique_idx),:]
                    y_new[int(t),:,:] = y_temp[np.sort(unique_idx),:]

                x_new_mean = np.mean(x_new, axis=0)
                y_new_mean = np.mean(y_new, axis=0)

            elif reverse_direction == True:
                y_len = np.unique(y_test[reference_condition,:], axis=0)
                x_len = x_test[reference_condition,:]

                x_new = np.zeros(shape=(n_trials, y_len.shape[0], x_len.shape[1]))
                y_new = np.zeros(shape=(n_trials, y_len.shape[0], y_len.shape[1]))

                for t, trial_number in enumerate(trials):
                    x_temp = x_test[np.where(test_trial_index == trial_number)[0],:]
                    y_temp = y_test[np.where(test_trial_index == trial_number)[0],:]

                    _, unique_idx = np.unique(y_temp, return_index=True, axis=0)
                    
                    x_new[int(t),:,:] = x_temp[np.sort(unique_idx),:]
                    y_new[int(t),:,:] = y_temp[np.sort(unique_idx),:]
                    
                    
                
                x_new_mean = np.mean(x_new, axis=0)
                y_new_mean = np.mean(y_new, axis=0)
                #print(y_new_mean[:,0])
                #print("---------------")

            #print(x_new.shape)
            #print(y_new.shape)
            if type(X_test) == bool:
                X_test = x_new_mean
                Y_test = y_new_mean

            elif type(X_test) != bool:
                X_test = np.vstack((X_test, x_new_mean))
                Y_test = np.vstack((Y_test, y_new_mean))
                
    
        if mode == "drive":
            np.save(f"temp/X_train_{idx}.npy", X_training)
            np.save(f"temp/X_test_{idx}.npy", X_test)
            np.save(f"temp/Y_train_{idx}.npy", Y_training)
            np.save(f"temp/Y_test_{idx}.npy", Y_test)
            #splits_X.append((X_training, X_test))
            #splits_Y.append((Y_training, Y_test))
            splits_X.append((f"temp/X_train_{idx}.npy", f"temp/X_test_{idx}.npy"))
            splits_Y.append((f"temp/Y_train_{idx}.npy", f"temp/Y_test_{idx}.npy"))

        elif mode == "ram":
            splits_X.append((X_training, X_test))
            splits_Y.append((Y_training, Y_test))
        
        idx_per_set.append(idx_subset)
    if give_indices == False:
        return splits_X, splits_Y
    elif give_indices != False:
        return splits_X, splits_Y, test_sets, idx_per_set


def select_stimulus_subset(X, Y, frame_index, trial_index, subset):
    """Select a subset of stimuli
    """
    subset_idx = [i for i in range(len(frame_index)) if frame_index[i] in subset]
    return X[subset_idx,:], Y[subset_idx,:], frame_index[subset_idx], trial_index[subset_idx]

