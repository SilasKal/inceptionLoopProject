import numpy as np
import scipy.ndimage as snd
import warnings
warnings.filterwarnings("ignore")

# Originally implemented by authors of  https://pubmed.ncbi.nlm.nih.gov/30349107/,  doi: 10.1038/s41593-018-0247-5
# and adapted for our downsampled data.

def lowhigh_normalize(frame, mask=None, sig_high=((100/5.4945054945054945) / 4), sig_low=(1 / 4)):
    ''' apply bandpass filter to frame
    specify mask and standard deviations sig_high (highpass) and sig_low (lowpass) of gaussian filters
    '''
    # recursion for frames
    if len(frame.shape)>2:
        result=np.empty_like(frame)
        for i in range(frame.shape[0]):
            result[i]=lowhigh_normalize(frame[i],mask, sig_high, sig_low)
        return result

    # recursion for complex images
    if np.iscomplexobj(frame):
        result=np.empty_like(frame)
        result.real=lowhigh_normalize(frame.real,mask,sig_high,sig_low)
        result.imag=lowhigh_normalize(frame.imag,mask,sig_high,sig_low)
        return result

    if mask is None:
        mask=np.ones(frame.shape,dtype=np.bool)
    data = np.copy(frame)
    m = np.zeros(mask.shape)
    m[mask] = 1.0
    m[np.logical_not(np.isfinite(frame))] = 0.
    data[np.logical_not(m)] = 0.
    m2 = np.copy(m)

    ## gaussian low pass
    low_mask = snd.gaussian_filter(m2,sig_low,mode='constant',cval=0)
    low_data = 1.*snd.gaussian_filter(data,sig_low,mode='constant',cval=0)/low_mask

    #### linear low pass
    ##t = np.linspace(-1,1,11)
    ##kernel = t.reshape(11,1)*t.reshape(1,11)
    ##low_mask = snd.convolve(m2, kernel, mode='constant', cval=0.0)
    ##low_data = snd.convolve(data, kernel, mode='constant', cval=0.0)/low_mask

    low_data[np.logical_not(m)] = 0
    high_mask = snd.gaussian_filter(m,sig_high,mode='constant',cval=0)
    highlow_data = low_data - 1.*snd.gaussian_filter(low_data,sig_high,mode='constant',cval=0)/high_mask
    highlow_data[np.logical_not(mask)]=np.nan

    return highlow_data