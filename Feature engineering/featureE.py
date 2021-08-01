import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler





def stat_feature(raw_data):
	from scipy.stats import skew, kurtosis
	def RMS(data):
		res = np.sum(data**2, axis=1)/data.shape[1]
		return res

	scaler = MinMaxScaler()
	raw_data = scaler.fit_transform(raw_data)

	var = np.var(raw_data, axis = 1)
	mean = np.mean(raw_data, axis = 1)
	# median = np.median(raw, axis = 1)
	sk = skew(raw_data, axis=1)
	kur = kurtosis(raw_data, axis =1)
	

	rms = RMS(raw_data)
	recit_mean = np.mean(np.abs(raw_data), axis=1)
	max_flu = (np.max(raw_data, axis=1) - np.min(raw_data, axis=1))
	form_factor = rms/mean
	peak_factor = np.max(raw_data, axis=1)/rms

	stat_fE = np.vstack((var, mean, sk, kur, rms, recit_mean, max_flu, form_factor,peak_factor)).T
	return stat_fE

def fft_ica(raw_data):
  from scipy.fft import fft
  from scipy.signal import find_peaks
	
  count = 0 # set initial counter
  for k in raw_data: # read data by line
    fft_co = fft(k)
    raw_peak, _ = find_peaks(k, threshold = 4)
    num_peaks = len(raw_peak) # count wave
    fft_co = (abs(fft_co)/fft_co.shape[0])

    length = fft_co.shape[0]
    fft_co = fft_co[0:(length//2)]


    peaks, _ = find_peaks(fft_co)
    
    coeff = fft_co[peaks][0:10] # fisrt 10 fft cofficients
    coeff = np.append(coeff, num_peaks) 
    res = coeff if count==0 else np.vstack((res, coeff))
    count+=1
# return a array with n*11 dimension
  return res

