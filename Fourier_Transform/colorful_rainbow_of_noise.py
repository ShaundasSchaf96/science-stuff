import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from IPython.display import Audio, display

# number of data points
npnts = int( 1e5 )

# vector of frequencies (normalized to Nyquist)
frex = np.linspace(0,1,int(npnts/2+1))

def white_noise_spectrum(amplitude, npnts):
    """
    Generate spectrum of white noise.

    Args:
        amplitude: amplitude of white noise

    Returns:
        magnitude of fourier coefficients from the fft function
    """
    noise = amplitude * np.random.randn(npnts)
    return abs(fft.fft(noise)/npnts)


def brown_noise_spectrum(amplitude, npnts):
    """
    Generate spectrum of brown noise.

    Args:
        amplitude: amplitude of brown noise

    Returns:
        magnitude of fourier coefficients from the fft function
    """
    noise = np.cumsum(amplitude*np.random.randn(npnts))
    return abs(fft.fft(noise)/npnts)


def pink_noise_spectrum(amplitude, npnts):
    """
    Generate spectrum of pink noise.

    Args:
        amplitude: amplitude of pink noise

    Returns:
        magnitude of fourier coefficients from the fft function
    """
    
    fourier_spec = np.zeros(npnts, dtype=complex)
    fourier_coeff_amp = 1/(frex + .01) + np.random.randn(int(npnts/2+1))**2*5
    fourier_coeff_phase = 2 * np.pi * np.random.rand(int(npnts/2+1))
    fourier_spec[:int(npnts/2+1)] = fourier_coeff_amp * np.exp(1j * fourier_coeff_phase)
    noise = amplitude * np.real(fft.ifft(fourier_spec))
    return abs(fft.fft(noise)/npnts)

def blueNoiseSpect(amplitude,npnts):
    """
    Generate spectrum of blue noise.

    Args:
        amplitude: amplitude of blue noise
        N: number of data points

    Returns:
        magnitude of fourier coefficients from the fft function
    """
    fourier_spec = np.zeros(npnts,dtype=complex)
    fourier_coeff_amp = np.linspace(1,3,int(npnts/2)+1) + np.random.randn(int(npnts/2+1))/5
    fourier_coeff_phs = 2*np.pi * np.random.rand(int(npnts/2+1))
    fourier_spec[:int(npnts/2+1)] = fourier_coeff_amp * np.exp(1j*fourier_coeff_phs)
    noise = amplitude * np.real(fft.ifft(fourier_spec))
    return abs(fft.fft(noise)/npnts)
    
    
def main():
    plt.plot(frex, white_noise_spectrum(1,npnts)[:len(frex)], color=[1,1,0])
    plt.plot(frex, brown_noise_spectrum(1,npnts)[:len(frex)], color=[.4,.2,.07])
    plt.plot(frex, pink_noise_spectrum(70,npnts)[:len(frex)], color=[1,0,1])
    plt.plot(frex, blueNoiseSpect(2000,npnts)[:len(frex)], color=[0,0,1])
    
    plt.ylim([0,.05])
    plt.ylabel('Amplitude (a.u.)')
    plt.xlabel('Frequency (fraction of Nyquist)')
    plt.title('Frequency domain')
    plt.show()
    
    # play audio
    display(Audio(brown_noise_spectrum(1,npnts), rate=44100))
    display(Audio(pink_noise_spectrum(70,npnts), rate=44100))
    display(Audio(blueNoiseSpect(2000,npnts), rate=44100))
    display(Audio(white_noise_spectrum(1,npnts), rate=44100))
    
if __name__ == "__main__":
    main()