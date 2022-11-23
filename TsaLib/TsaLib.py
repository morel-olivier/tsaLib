#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import pyfda.filter_designs.common as fda	# use version 0.5.3
# TODO: find a way to force the usage of V0.5.3

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

	à tester:
	---------

	ndft plus grand, plus petit est = à len(x)

	à améliorer
	===========

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

def firLowPassDesign(amps,rp,rs,cutoff,trans_width,Fs):
	"""Design of lowpass filter

	Parameters
	----------
	amps : array-like
		desired gain in each of the specified bands [-]
		Should be [1,0]
	rp : float
		ripple in the passband [dB]
	rs : float
		ripple in the stopband [dB]
	cutoff : float
		passband frequency cut [Hz]
	trans_width : float
		transition width between passband and stop band [Hz]
	Fs : float
		frequency sampling [Hz]

	Returns
	-------
	num : array-like
		coefficients of the filter
	"""

	# Convert deviation to linear units
	rips = [(10**(rp/20)-1)/(10**(rp/20)+1), 10**(-rs/20)]
	
	# Determine the filter order
	freqs = [cutoff, cutoff + trans_width] # [Fpass2, Fstop2]
	(n,f0,a0,w) = fda.remezord(freqs, amps, rips, Fs)
	
	# Determine the filter coefficients
	edges = [0, cutoff, cutoff + trans_width, Fs/2] # [0, Fpass2, Fstop2, Fs/2]
	num = signal.remez(n,edges,amps,w, fs = Fs) 
	# or num = signal.remez(n,f0,amps,w)
	return num

def firHighPassDesign(amps,rp,rs,cutoff,trans_width,Fs):
	"""Design of high pass filter

	Parameters
	----------
	amps : array-like
		desired gain in each of the specified bands [-]
		Should be [0,1]
	rp : float
		ripple in the passband [dB]
	rs : float
		ripple in the stopband [dB]
	cutoff : float
		passband frequency cut [Hz]
	trans_width : float
		transition width between passband and stop band [Hz]
	Fs : float
		frequency sampling [Hz]

	Returns
	-------
	num : array-like
		coefficients of the filter
	"""
	rips = [10**(-rs/20), (10**(rp/20)-1)/(10**(rp/20)+1)] # original rips = [(10**(rp/20)-1)/(10**(rp/20)+1), 10**(-rs/20)]

	# Determine the filter order
	(n,fzero,a0,w) = fda.remezord([cutoff-trans_width, cutoff], amps, rips, Fs)

	if n%2 == 0:
		n+=1

	# Determine the filter coefficients
	edges = [0, cutoff-trans_width, cutoff, Fs/2] # [0, Fpass2, Fstop2, Fs/2]
	num = signal.remez(n,edges,amps,w, fs = Fs)
	return num

def firBandPassDesign(amps,rp,rs,Fpass1, Fpass2,trans_width, Fs):
	"""Design of band pass filter

	Parameters
	----------
	amps : array-like
		desired gain in each of the specified bands [-]
		Should be [0, 1, 0]
	rp : float
		ripple in the passband [dB]
	rs : float
		ripple in the stopband [dB]
	Fpass1 : float
		first frequency cut [Hz]
	Fpass2 : float
		second frequency cut [Hz]
	trans_width : float
		transition width between passband and stop band [Hz]
	Fs : float
		frequency sampling [Hz]

	Returns
	-------
	num : array-like
		coefficients of the filter
	"""
	# Convert deviation to linear units
	rips = [(10**(rp/20)-1)/(10**(rp/20)+1), 10**(-rs/20), (10**(rp/20)-1)/(10**(rp/20)+1)]

	# Determine the filter order
	freqs = [Fpass1-trans_width , Fpass1, Fpass2, Fpass2+trans_width] # [Fpass2, Fstop2]
	n,fzero,a0,w = fda.remezord(freqs, amps, rips, Fs)

	# Determine the filter coefficients
	edges = [0, freqs[0], freqs[1], freqs[2], freqs[3], Fs/2] # [0, Fpass2, Fstop2, Fs/2]
	num = signal.remez(n,edges,amps,w, fs = Fs)
	return num

def firBandStopDesign(amps,rp,rs,Fpass1, Fpass2, trans_width, Fs):
	"""Design of band stop filter

	Parameters
	----------
	amps : array-like
		desired gain in each of the specified bands [-]
		Should be [1, 0, 1]
	rp : float
		ripple in the passband [dB]
	rs : float
		ripple in the stopband [dB]
	Fpass1 : float
		first frequency cut [Hz]
	Fpass2 : float
		second frequency cut [Hz]
	trans_width : float
		transition width between passband and stop band [Hz]
	Fs : float
		frequency sampling [Hz]

	Returns
	-------
	num : array-like
		coefficients of the filter
	"""
	# Convert deviation to linear units
	rips = [(10**(rp/20)-1)/(10**(rp/20)+1), 10**(-rs/20), (10**(rp/20)-1)/(10**(rp/20)+1)]

	# Determine the filter order
	freqs = [Fpass1, Fpass1+trans_width, Fpass2-trans_width, Fpass2] # [Fpass2, Fstop2]
	n,fzero,a0,w = fda.remezord(freqs, amps, rips, Fs)

	# Determine the filter coefficients
	edges = [0, freqs[0], freqs[1], freqs[2], freqs[3], Fs/2] # [0, Fpass2, Fstop2, Fs/2]
	num = signal.remez(n,edges,amps,w, fs = Fs)
	return num

def displayBode(Fs, b, a = 1):
	"""Display the bode of a transfert function.

	This is not a real bode as there no logarithmic axes.

	Parameters
	----------
	Fs : float
		sampling frequency
	b : array-like
		denominator of the transfer function
	a : array-like, optional
		minator of the transfer function, by default 1

	Returns
	-------
	nothing
	"""
	# Frequency analysis
	w, H = signal.freqz(b,a)
	HdB = 20*np.log10(abs(H))
	Hphase = np.unwrap(np.angle(H))

	# Display
	plt.figure(figsize=(10,6))
	plt.subplot(2,1,1)
	plt.plot(w*Fs/2/np.pi,HdB)
	plt.xlim((0,Fs/2))
	plt.title('Low pass filter : Magnitude diagram')
	plt.xlabel('f [Hz]')
	plt.ylabel('|H| [dB]')
	plt.grid(True)
	plt.subplot(2,1,2)
	plt.plot(w*Fs/2/np.pi,Hphase)
	plt.xlim((0,Fs/2))
	plt.title('Phase diagram')
	plt.xlabel('f [Hz]')
	plt.ylabel('angle(H) [rad]')
	plt.grid(True)

	plt.tight_layout()
	plt.show()

def firFilter(x, B):
	y = np.zeros(len(x))
	for n in range(len(x)):
		for k in range(len(B)):
			if n-k >= 0:
				y[n] += x[n-k] * B[k]
	return y

class cIIrFilter:
	def __init__(self, gains, B, A):
		self.gains = gains
		self.B = B
		self.A = A
		self.xPrev = np.zeros(len(B))
		self.yPrev = np.zeros(len(A))

	def display(self):
		print(self.gains)
		print(self.B)
		print(self.A)
		print(self.xPrev)

	def filt(self, x):
		self.xPrev[1:len(self.B)] = self.xPrev[0:len(self.B)-1]
		self.xPrev[0] = x * self.gains
		self.yPrev[1:len(self.A)] = self.yPrev[0:len(self.A)-1]
		y = np.dot(np.transpose(self.xPrev), self.B) - np.dot(np.transpose(self.yPrev[1:]), self.A[1:])
		self.yPrev[0] = y
		return y

def chirpCreate(ampl,fs,f0,f1,t0,t1):
	"""Generate a chirp signal

	Parameters
	----------
	ampl : float
		Amplitude of the chirp
	fs : float
		Sampling rate
	f0 : float
		First frequency
	f1 : float
		Last frequency
	t0 : float
		Start time
	t1 : float
		End time

	Returns
	-------
	t : ndarray
		Time vector
	f : ndarray
		Frequency vector
	chirp : ndarray
		Chirp signal
	"""
	t = np.linspace(t0,t1, (t1-t0)*fs)
	f = np.linspace(f0, f1, len(t))
	beta = (f1-f0)/(t1-t0)

	phi = ampl*(np.pi* beta * t**2 + 2*np.pi*f0*t)
	chirp = np.sin(phi)

	return t, f,chirp