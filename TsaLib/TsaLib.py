#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Do not care about maxima on the first w and last w values
#safe in theory
def findFirstPeak(x,w=1):
	"""Find first peak in signal

	Args:
		x (array_like): Signal
		w (int, optional): Size of the window. Th maximum MUST be the absolute maximum in a window of size w around it. Defaults to 1.

	Returns:
		lagMax:	integer
				Indice of the first peak
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
	"""Compute unbiased auto correlation

	Args:
		x (_type_): _description_
		kMin (_type_): _description_
		kMax (_type_): _description_

	Returns:
		_type_: _description_
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
	"""Compute biased auto correlation

	Args:
		x (_type_): _description_
		kMin (_type_): _description_
		kMax (_type_): _description_

	Returns:
		_type_: _description_
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
	lagMax = kMin
	xMax = 0
	if kMin >= kMax or kMin < 0 or kMax > len(x)-1:	# error cases
		return lagMax, xMax
	lagMax = np.argmax(x[kMin:kMax])+kMin
	#lagMax = np.argmax(x[kMin:kMax+1])+kMin
	xMax = x[lagMax]
	return lagMax, xMax

def findPeak(x, kMin, kMax):
	return findMax(x, kMin, kMax)

def dummPython(x):
	return x


def correlationCoefficient(x,y):
	x = x - np.mean(x)
	y = y - np.mean(y)	# centralization
	r = np.correlate(x,y, mode='valid')
	r /= np.sqrt(np.correlate(x,x, mode='valid')*np.correlate(y,y, mode='valid'))
	return r

# TODO: add normalized cross correlation

# TODO: use standard function
def computeDFT(x, ndft, hamming=False):
	N = len(x)
	if N < ndft:
		x = np.append(x, np.zeros(ndft-N))
	w = np.arange(0,1, 1/ndft)
	k = np.arange(0, ndft)
	dft = np.zeros(ndft, dtype = 'complex_')
	
	#for n in range(ndft):
	#	dft[n] = np.sum(np.dot(x, np.exp((-1j*2*np.pi*n*k)/ndft)))
	return dft, w

def computePSD(x, ndft, hamming=False):
	dft, w = computeDFT(x,ndft, hamming=hamming)
	psd = (np.abs(dft)**2)
	psd /= ndft
	return psd, dft, w

def pitch2Tone(freq):
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
		signal (array_like): Signal used to find the tone
		Fs (real): Sampling Frequency in [Hz]

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
