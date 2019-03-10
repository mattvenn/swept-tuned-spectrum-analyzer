import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# sampling rate
fs = float(10000)
time = 0.1 # s

samples = np.linspace(0, time, int(fs*time), endpoint=False)

input_freq1 = 4000  # Hz
print("input signal 1 freq %d Hz" % input_freq1)
in_signal1 = np.sin(2 * np.pi * input_freq1 * samples) * 0.5

input_freq2 = 4000  # Hz
print("input signal 2 freq %d Hz" % input_freq2)
in_signal2 = np.sin(2 * np.pi * input_freq2 * samples) * 0.5

def plots():
    # do all the plots
    num_plots = 4

    # plot input signal
    fig = plt.figure()
    ax = fig.add_subplot(num_plots,1,1)
    ax.set_title("input signal 1&2 at %d and %d Hz. FS = %d Hz" % (input_freq1, input_freq2, fs))
    ax.set_xlabel('time')
    ax.set_ylabel('amp')
    ax.plot(samples, in_signal1, 'r')
    ax.plot(samples, in_signal2, 'y')

    # plot mixed signal
    mixed = in_signal1 * in_signal2

    ax = fig.add_subplot(num_plots,1,3)
    ax.set_title("1 * 2")
    ax.set_xlabel('time')
    ax.set_ylabel('amp')
    ax.plot(samples, mixed, 'b')

    #fft
    n = len(samples)
    k = np.arange(n/2)
    T = n/fs
    frq = k/T # two sides frequency range

    Y = np.fft.fft(mixed)/n # fft computing and normalization
    Y = Y[range(int(n/2))]

    ax = fig.add_subplot(num_plots,1,4)
    ax.set_title("fft of mixed")
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('|Y(freq)|')
    ax.plot(frq, abs(Y), 'b')

    ax = fig.add_subplot(num_plots,1,2)
    ax.set_title("fft of 1 & 2")
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('|Y(freq)|')

    Y = np.fft.fft(in_signal1)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    ax.plot(frq, abs(Y), 'r')

    Y = np.fft.fft(in_signal2)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    ax.plot(frq, abs(Y), 'y')

    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plots()
