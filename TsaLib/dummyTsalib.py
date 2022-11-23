def dummyComputeDFT(x, ndft, hamming=False):
	"""Compute Discrete Fourrier Transform.

	Args:
		x (array-like): Signal on wich the DFT will be perform.

		ndft (integer): Number of samples in the dft.

		hamming (bool, optional): hamming. Defaults to False.

	Returns:
		dft: ndarray
			Complex values of the dft.
		w: ndarray
			Normalized frequencies.
	"""
	N = len(x)
	if N < ndft:
		x = np.append(x, np.zeros(ndft-N))
	w = np.arange(0,1, 1/ndft)
	k = np.arange(0, ndft)
	dft = np.zeros(ndft, dtype = 'complex_')
	
	for n in range(ndft):
		dft[n] = np.sum(np.dot(x, np.exp((-1j*2*np.pi*n*k)/ndft)))
	return dft, w

def dummyChirpCreate(ampl,fs,f0,f1,t0,t1):
	"""Create a chirp signal in a not efficient way

	Parameters
	----------
	ampl : integer
		Amplitude of the chirp
	fs : integer
		Sampling frequency
	f0 : integer
		Start frequency
	f1 : integer
		End frequency
	t0 : integer
		Start time
	t1 : integer
		End time

	Returns
	-------
	t : ndarray
		Time vector of the chirp.
	chirp : ndarray
		Chirp signal.
	"""
	t = np.linspace(t0,t1, (t1-t0)*fs)
	beta = (f1-f0)/(t1-t0)

	phi = ampl*(np.pi* beta * t**2 + 2*np.pi*f0*t)
	chirp = np.sin(phi)

	return t,chirp