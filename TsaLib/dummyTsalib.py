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