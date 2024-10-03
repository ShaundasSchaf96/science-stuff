import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc

# Set animation defaults
rc('animation', html='jshtml')

# Parameters
FREQ = 5  # Hz
FWHM = 0.5
SRATE = 500  # Hz
TIME = np.arange(-2 * SRATE, 2 * SRATE) / SRATE
NPPTS = len(TIME)

def create_complex_morlet_wavelet(time, freq=FREQ, fwhm=FWHM, phs=0):
    """
    Create a complex Morlet wavelet.

    Parameters:
        time (np.ndarray): Time vector.
        freq (float): Frequency of the wavelet (Hz).
        fwhm (float): Full-width at half maximum.
        phs (float): Phase shift (radians).

    Returns:
        np.ndarray: Complex Morlet wavelet.
    """
    sinepart = np.exp(1j * (2 * np.pi * freq * time + phs))
    gausspart = np.exp((-4 * np.log(2) * time ** 2) / (fwhm ** 2))
    return sinepart * gausspart

def aframe(phs):
    """
    Update the plots for the animation.

    Parameters:
        phs (float): Phase shift to be applied.

    Returns:
        tuple: Updated plot lines.
    """
    wavelet = create_complex_morlet_wavelet(TIME, FREQ, FWHM, phs)
    plth1.set_ydata(np.real(wavelet))
    plth2.set_ydata(np.imag(wavelet))
    return plth1, plth2

def create_animation():
    """
    Create and display the animation of the Morlet wavelet.
    """
    fig, ax = plt.subplots(1, figsize=(12, 6))

    global plth1, plth2  # Use global variables for plot lines
    plth1, = ax.plot(TIME, np.zeros(NPPTS), label='Real Part')
    plth2, = ax.plot(TIME, np.zeros(NPPTS), label='Imaginary Part')
    ax.set_ylim([-1, 1])
    ax.set_title('Complex Morlet Wavelet Animation')
    ax.legend()

    phases = np.linspace(0, 2 * np.pi - 2 * np.pi / 10, 10)
    ani = animation.FuncAnimation(fig, aframe, phases, interval=50, repeat=True)
    plt.show()

def main():
    create_animation()

if __name__ == '__main__':
    main()
