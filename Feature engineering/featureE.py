import numpy as np
import pandas as pd
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

	fft_co = fft(raw_data)
	fft_co = (abs(fft_co)/fft_co.shape)

	length = fft_co.shape[0]
	fft_co = fft_co[0:(length//2)]

	peaks, _ = find_peaks(fft_co, threshold=4)
	
	coeff = fft_co[peaks]
	return fft_co, peaks


