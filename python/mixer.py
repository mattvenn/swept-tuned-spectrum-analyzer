import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LW = 0.5
# sampling rate
fs = float(10000)
time = 0.1 # s
input_freq2 = 2000 # hz
start_freq = 0
stop_freq = int(fs/2)
step_freq = 50

samples = np.linspace(0, time, int(fs*time), endpoint=False)

def plots(input_freq1, sig1,sig2,mixed,fft1,fft2,fftmixed):
    print("input signal 1 freq %d Hz" % input_freq1)
    in_signal1 = np.sin(2 * np.pi * input_freq1 * samples) * 0.5

    print("input signal 2 freq %d Hz" % input_freq2)
    in_signal2 = np.sin(2 * np.pi * input_freq2 * samples) * 0.5

    sig1.set_data(samples, in_signal1)
    sig2.set_data(samples, in_signal2)

    # plot mixed signal
    mixed_sig = in_signal1 * in_signal2

    mixed.set_data(samples, mixed_sig)

    #fft
    n = len(samples)
    k = np.arange(n/2)
    T = n/fs
    frq = k/T # two sides frequency range

    Y = np.fft.fft(mixed_sig)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    fftmixed.set_data(frq, abs(Y))

    Y = np.fft.fft(in_signal1)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    fft1.set_data(frq, abs(Y))

    Y = np.fft.fft(in_signal2)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    fft2.set_data(frq, abs(Y))
    return sig1, sig2, mixed, fft1, fft2, fftmixed

def setup_plots():
    # do all the plots
    num_plots = 4

    # plot input signal
    fig = plt.figure()
    ax1 = fig.add_subplot(num_plots,1,1)
    ax1.set_title("input signal 1&2 at %d and %d->%d Hz. FS = %d Hz" % (input_freq2, start_freq, stop_freq, fs))
    ax1.set_xlabel('time')
    ax1.set_ylabel('amp')
    ax1.set_xlim(0, time)
    ax1.set_ylim(-1, 1)
    ax1.grid()

    sig1, = ax1.plot([],[], 'r', linewidth=LW)
    sig2, = ax1.plot([],[], 'y', linewidth=LW)

    ax2 = fig.add_subplot(num_plots,1,3)
    ax2.set_title("1 * 2")
    ax2.set_xlabel('time')
    ax2.set_ylabel('amp')
    ax2.set_xlim(0, time)
    ax2.set_ylim(-1, 1)
    ax2.grid()
    mixed, = ax2.plot([], [], 'b', linewidth=LW)


    ax3 = fig.add_subplot(num_plots,1,4)
    ax3.set_title("fft of mixed")
    ax3.set_xlabel('Freq (Hz)')
    ax3.set_ylabel('|Y(freq)|')
    ax3.set_xlim(0, fs/2)
    ax3.set_ylim(0, 0.10)
    ax3.grid()
    fftmixed, = ax3.plot([], [], 'b', linewidth=LW)

    ax4 = fig.add_subplot(num_plots,1,2)
    ax4.set_title("fft of 1 & 2")
    ax4.set_xlabel('Freq (Hz)')
    ax4.set_ylabel('|Y(freq)|')
    ax4.set_xlim(0, fs/2)
    ax4.set_ylim(0, 0.25)
    ax4.grid()
    fft1, = ax4.plot([], [], 'r', linewidth=LW)
    fft2, = ax4.plot([], [], 'y', linewidth=LW)

    return fig,sig1, sig2, mixed, fft1, fft2, fftmixed

def freq_range():
    return iter( range(start_freq, stop_freq, step_freq))

if __name__ == '__main__':
    fig, sig1, sig2, mixed, fft1, fft2, fftmixed = setup_plots()
#    plots(ax1,ax2,ax3,ax4,500,800)
    line_ani = animation.FuncAnimation(fig, plots, freq_range, repeat=True, fargs=(sig1, sig2, mixed, fft1, fft2, fftmixed), interval=50, blit=False)
    plt.show()
