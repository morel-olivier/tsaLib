#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Do not care about maxima on the first w and last w values
#safe in theory
def findFirstPeak(x,w=1):
	"""Find first peak in signal.

	Args:
		x (array_like): Signal.

		w (int, optional): Size of the window. The maximum MUST be the absolute maximum in a window of size w around it. Defaults to 1.

	Returns:
		lagMax:	integer
				Index of the first peak.py
	"""
	lagMax = -1
	i = w
	while lagMax == -1 and i < len(x) -w:
		#print("i = ", i)
		#print("x = ", x[i-w:i+w+1])
		if np.argmax(x[i-w:i+w+1]) == w:
			lagMax = i
		i+=1
	return lagMax

# not safe
def unbiasedAutoCorrelation(x, kMin, kMax):
	"""Compute unbiased auto correlation.

	Args:
		x (array_like): Signal to be autocorrelated.

		kMin (integer): Start index of correlation.

		kMax (integer): End index of correlation.

	Returns:
		lags:	ndarray
				Index of the correlation (from kMin to Kmax with a step of 1).
		rxx:	ndarray
				Unbiased auto correlation of x.
		
	"""
	lags = np.arange(kMin,kMax +1)	# WARNING: add 1 to kMax besause np.arange stop 1 before the value it was given
	rxx = np.zeros(kMax - kMin + 1)
	for i in lags:
		#print(i)
		#print(x[:i])
		rxx[i] = np.dot(x[:kMax-i+1], x[i:])/(len(x) - abs(i))
	return lags,rxx

# not safe and not optimized
def biasedAutoCorrelation(x, kMin, kMax):	# TODO: improve security
	"""Compute biased auto correlation.

	Args:
		x (ndarray): Signal to be correlated

		kMin (integer): Start index of correlation.

		kMax (integer): End index of correlation.

	Returns:
		lags:	ndarray
				Index of the correlation (from kMin to Kmax with a step of 1).
		
		rxx:	ndarray
				Biased auto correlation of x.
	"""
	lags = np.arange(kMin,kMax +1)
	rxx = np.zeros(kMax - kMin + 1)
	for k in range(kMin, kMax+1):
		for n in range(0, len(x)):
			if n+k >= 0 and n+k < len(x):
				rxx[k] += x[n]*x[n+k]
		#rxx[k]/=len(x)
	rxx = rxx/len(x)
	return lags,rxx


# default of lagMax = kMin
# default of xMax = 0
def findMax(x, kMin, kMax):
	"""Find the maximum of a signal.

	Args:
		x (array-like): Signal.

		kMin (integer): Start index of search.

		kMax (integer): End index of search.

	Returns:
		lagMax: integer
				Index of the greatest value in x.
				Default to kMin.

		xMax:	integer
				Maximum value found in x.
				Default to 0.
	"""
	lagMax = kMin
	xMax = 0
	if kMin >= kMax or kMin < 0 or kMax > len(x)-1:	# error cases
		return lagMax, xMax
	lagMax = np.argmax(x[kMin:kMax])+kMin
	#lagMax = np.argmax(x[kMin:kMax+1])+kMin
	xMax = x[lagMax]
	return lagMax, xMax

def correlationCoefficient(x,y):
	"""Compute the correlation coefficient between two signals.

	Args:
		x (array-like): Signal to correlate.

		y (array-like): Signal to correlate.

	Returns:
		r: integer
			Correlation coefficient between x and y.
	"""
	# centralization
	x = x - np.mean(x)
	y = y - np.mean(y)

	r = np.correlate(x,y, mode='valid')
	r /= np.sqrt(np.correlate(x,x, mode='valid')*np.correlate(y,y, mode='valid'))
	return r

# TODO: add normalized cross correlation

def computeDFT(x, ndft, hamming=False):
	"""Compute Discrete Fourrier Transform.

	Args:
		x: array-like
			Signal on wich the DFT will be perform.

		ndft: integer
			Number of samples in the dft.
			If ndft is greater than the length of x then zeros are appended to the end of x (zero padding).
			If ndft is smaller than the length of x then x is truncated to have a length o ndft.
			This is

		hamming: bool, optional
			hamming. Defaults to False.

	Returns:
		dft: ndarray
			Complex values of the dft.
		w: ndarray
			Normalized frequencies.
	
	y[1:3:2] sort les valeurs de 1 à 3 par pas de 1

	## à tester:
	ndft plus grand, plus petit est = à len(x)

	## à améliorer
	permetre de renvoyer un tableau de fréquence bipolair (-Fs/2 à Fs/2) au lieu de 0 à Fs
	add normalization parameter
	"""
	N = len(x)
	w = np.arange(0,1, 1/ndft)
	dft = np.fft.fft(x, ndft)
	return dft, w

def computePSD(x, ndft, hamming=False):
	"""Compute Power Spectral Density

	Args:
		x (array-like): signal on wich the DFT and the PSD.
		ndft (integer): Number of samples in the dft.
		hamming (bool, optional): hamming. Defaults to False.

	Returns:
		psd: ndarray
			Power Spectral Density of x.
		dft: ndarray
			Complex values of the dft.
		w: ndarray
			Normalized frequencies.

	"""
	dft, w = computeDFT(x,ndft, hamming=hamming)
	psd = (np.abs(dft)**2)
	psd /= ndft
	return psd, dft, w

def pitch2Tone(freq):
	"""pitch to tone

	Parameters
	----------
	freq : float
		frequecy used to determine the tone.

	Returns
	-------
	tone: string
		Letter of the tone.
	o: integer
		octave of the tone.
	t: integer
		index of the tone
	"""
	tone=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

	fRef = 440
	octRef = 3

	a = freq/fRef
	n = 45 + np.log10(a)/np.log10(2**(1/12))	# warning root
	n = round(n)
	o = n//12
	t = n%12

	return tone[t],o,t

def signal2Tone(signal, Fs):
	"""Signal to tone

	Args:
		signal (array_like): Signal used to find the tone.
		
		Fs (real): Sampling Frequency in [Hz].

	Returns:
		Fy 		:	real
					Frequency of th signal.
		tone	:	string
					Tone of the signal in letter.
		octave 	:	integer
					Octave of the signal (I don't rememeber the norm used to calculate, it may on lower or higher).
		t 		:	integer
					Tone of the signal but in integer form.
	"""
	rxx = np.correlate(signal, signal, 'same')
	firstPeak = findFirstPeak(rxx)
	secondPeak = findFirstPeak(rxx, w=firstPeak+1)
	Fy = Fs/(secondPeak-firstPeak)
	tone, octave, t = pitch2Tone(Fy)
	return Fy, tone, octave, t

def spectogramme(X, nfft):
	X = 0.0
	return X
