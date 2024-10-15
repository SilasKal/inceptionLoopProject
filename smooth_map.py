#!/usr/bin/python

'''
filter functions
'''

__author__ = 'Bettina Hein'
__email__ = 'hein@fias.uni-frankfurt.de'

import numpy as np
import scipy.ndimage as snd
import mustela.conf as c
import warnings
warnings.filterwarnings("ignore")


def low_normalize(frame, mask=None, sigma=c.sigma_low):
	''' apply lowpass filter to frame
	specify mask and standard deviation sigma of gaussian filter
	'''
	# recursion for frames
	if len(frame.shape)>2:
		result=np.empty_like(frame)
		for i in range(frame.shape[0]):
			result[i]=low_normalize(frame[i],mask,sigma)
		return result
		
	# recursion for complex images
	if np.iscomplexobj(frame):
		result=np.empty_like(frame)
		result.real=low_normalize(frame.real,mask,sigma)
		result.imag=low_normalize(frame.imag,mask,sigma)
		return result
		
	if mask is None:
		mask=np.ones(frame.shape,dtype=np.bool)

	data = np.copy(frame)
	m = np.zeros(mask.shape)
	m[mask] = 1.0
	m[~np.isfinite(frame)] = 0.
	data[np.logical_not(m)] = 0.
	normalized_data = 1.*snd.gaussian_filter(data,sigma,mode='constant',cval=0)/snd.gaussian_filter(m,sigma,mode='constant',cval=0)
	normalized_data[np.logical_not(mask)]=np.nan
	return normalized_data

def high_normalize(frame, mask=None, sigma=c.sigma_high):
	''' apply highpass filter to frame
	specify mask and standard deviation sigma of gaussian filter
	'''
	# recursion for frames
	if len(frame.shape)>2:
		result=np.empty_like(frame)
		for i in range(frame.shape[0]):
			result[i]=high_normalize(frame[i],mask,sigma)
		return result
		
	# recursion for complex images
	if np.iscomplexobj(frame):
		result=np.empty_like(frame)
		result.real=high_normalize(frame.real,mask,sigma)
		result.imag=high_normalize(frame.imag,mask,sigma)
		return result
		
	if mask is None:
		mask=np.ones(frame.shape,dtype=np.bool)

	data = np.copy(frame)
	m = np.zeros(mask.shape)
	m[mask] = 1.0
	m[~np.isfinite(frame)] = 0.
	data[np.logical_not(m)] = 0.
	normalized_data = data - 1.*snd.gaussian_filter(data,sigma,mode='constant',cval=0)/snd.gaussian_filter(m,sigma,mode='constant',cval=0)
	normalized_data[np.logical_not(mask)]=np.nan
	return normalized_data
	

def high_normalize_dew(frame, mask=None):
	''' apply highpass filter with 2d median filter (800microns)
	followed by thresholding at th 20th percentile
	specify mask and standard deviation sigma of gaussian filter
	'''

	# recursion for frames
	if len(frame.shape)>2:
		result=np.empty_like(frame)
		for i in range(frame.shape[0]):
			result[i]=high_normalize_dew(frame[i],mask)
		return result
		
	# recursion for complex images
	if np.iscomplexobj(frame):
		result=np.empty_like(frame)
		result.real=high_normalize_dew(frame.real,mask)
		result.imag=high_normalize_dew(frame.imag,mask)
		return result
		
	if mask is None:
		mask=np.ones(frame.shape,dtype=np.bool)

	
	data = np.copy(frame)
	m = np.zeros(mask.shape)
	m[mask] = 1.0
	m[-np.isfinite(frame)] = 0.
	#data[np.logical_not(m)] = 0.
	
		
	sigma_mic=800	#microns
	sigma = (1.*data.shape[0]/270)*sigma_mic/c.px_to_ums
	
	normalized_data = data - 1.*snd.filters.median_filter(data,sigma,mode='constant',cval=0)
	normalized_data[np.logical_not(mask)]=np.nan
	return normalized_data
	


def lowhigh_normalize(frame, mask=None, sig_high=c.sigma_high, sig_low=c.sigma_low):
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



if __name__=='__main__':
	from mustela.datasets import descriptions
	from mustela.tools import loading
	from bettina.tools import get_map
	import matplotlib.pyplot as plt
	
	ferret=1581
	dataset_opm = descriptions.get_grating_best_opm(ferret,'early')
	roi = loading.load_roi(dataset_opm,'transformation')
	
	m0 = get_map.get_map([dataset_opm],None,None)
	mr = get_map.get_map([dataset_opm],None,roi)
	
	fn = high_normalize(m0,mask=roi)
	f0 = high_normalize_dew(m0,mask=roi)
	fr = high_normalize_dew(mr,mask=None)
	
	fig=plt.figure()
	ax = fig.add_subplot(221)
	ax.imshow(np.real(m0),interpolation='nearest',cmap='binary')
	
	ax = fig.add_subplot(222)
	ax.imshow(np.real(fn),interpolation='nearest',cmap='binary')
	
	ax = fig.add_subplot(223)
	ax.imshow(np.real(f0),interpolation='nearest',cmap='binary')
	
	ax = fig.add_subplot(224)
	ax.imshow(np.real(fr),interpolation='nearest',cmap='binary')
	
	plt.show()
	


