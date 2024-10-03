import numpy as np
import matplotlib.pyplot as plt

def generate_time_series(alpha, N):
  """
  Generate the time series with temporal weighting and random noise.

  Args:
      alpha: the temporal weighting vector
      N: the number of time points

  Returns:
      x: the time series
  """
  k = len(alpha)
  x = np.zeros(N)
    
  # Generate the time series based on the temporal weighting vector
  for i in range(k, N):
    x[i] = sum(alpha * x[i-k:i]) + np.random.randn()

  # Add a sine wave to the time series
  x += np.sin(np.linspace(0, 10 * np.pi, N))
  return x

def manual_fourier_transform(x, N):
  """
  Compute the manual Fourier Transform.

  Args:
      x: the time series
      N: the number of time points

  Returns:
      fourier_coeff: the Fourier coefficients
  """
  time_vector = np.arange(0, N) / N
  fourier_coeff = np.zeros(N, dtype=complex)
    
  for f in range(N):
    complex_sine_wave = np.exp(-1j * 2 * np.pi * f * time_vector)
    fourier_coeff[f] = np.dot(complex_sine_wave, x)
        
  return fourier_coeff

def fast_fourier_transform(x):
  """
  Compute the fast Fourier Transform.

  Args:
      x: the time series

  Returns:
      fast_fourier_coeff: The Fourier coefficients using the FFT
  """
  fast_fourier_coeff = np.fft.fft(x)
  return fast_fourier_coeff

def plot_fourier_transforms(frequencies_vector, manual_coeff, fft_coeff):
  """
  Plot the Fourier Transforms.

  Args:
      frequencies_vector: frequencies vector
      manual_coeff: the Fourier coefficients computed with the manual Fourier Transform
      fft_coeff: the Fourier coefficients computed with the fast Fourier Transform
  """
  plt.plot(frequencies_vector, np.abs(manual_coeff[:len(frequencies_vector)]), label='Manual Fourier Transform')
  plt.plot(frequencies_vector, np.abs(fft_coeff[:len(frequencies_vector)]), 'ro', label='Fast Fourier Transform')
  plt.xlabel('Frequency (fraction of Nyquist)')
  plt.ylabel('Amplitude (a.u.)')
  plt.legend()
  plt.show()

def main():
  alpha = np.array([-0.6, 0.9])
  N = 200  # random chosen number of time points
    
  x = generate_time_series(alpha, N)
    
  fourier_coeff_manual = manual_fourier_transform(x, N)
  fourier_coeff_fft = fast_fourier_transform(x)
    
  frequencies_vector = np.linspace(0, 1, int(N / 2 + 1))
  plot_fourier_transforms(frequencies_vector, fourier_coeff_manual, fourier_coeff_fft)

if __name__ == "__main__":
  main()
