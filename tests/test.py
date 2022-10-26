import TsaLib.TsaLib as tlb
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	x = np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
	mydft, myw = tlb.computeDFT(x, len(x))
	
	dft = np.fft.fft(x, len(x))
	#dft = np.fft.fftshift(dft)
	w = np.fft.fftfreq(len(x))
	w = np.fft.fftshift(w)

	print(len(dft))
	print(len(mydft))

	plt.figure()
	plt.plot(myw,abs(dft), 'o-')
	plt.plot(myw, abs(mydft), 'x-')
	plt.show()
