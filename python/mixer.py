import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LW = 0.5
# sampling rate
fs = float(10000)
time = 0.01 # s
input_freq1 = 1000 # hz
input_freq2 = 1300 # hz
start_freq = 0
stop_freq = int(fs/2)
step_freq = 100
IF_filt_freq = 100

samples = np.linspace(0, time, int(fs*time), endpoint=False)

print("input signal1  freq %d Hz" % input_freq1)
print("input signal2  freq %d Hz" % input_freq2)
in_signal = np.sin(2 * np.pi * input_freq1 * samples) * 0.5 + np.sin(2 * np.pi * input_freq2 * samples) * 0.5

# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass(lowcut, highcut, fs, order=5, btype='bandpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if btype == 'bandpass':
        b, a = signal.butter(order, [low, high], btype=btype)
    elif btype == 'low':
        b, a = signal.butter(order, [low], btype=btype)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

b, a = butter_bandpass(IF_filt_freq, IF_filt_freq, fs, btype="low")

def plots(sweep_freq, sig_LO,sig_input,mixed,fft1,fft2,fftmixed, filtered, filtered_time):
    print("input signal 1 freq %d Hz" % sweep_freq)
    LO = np.sin(2 * np.pi * sweep_freq * samples) * 0.5


    sig_LO.set_data(samples, LO)
    sig_input.set_data(samples, in_signal)

    # plot mixed signal
    mixed_sig = LO * in_signal

    mixed.set_data(samples, mixed_sig)

    #fft
    n = len(samples)
    k = np.arange(n/2)
    T = n/fs
    frq = k/T # two sides frequency range

    Y = np.fft.fft(mixed_sig)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    fftmixed.set_data(frq, abs(Y))

    Y = np.fft.fft(LO)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    fft1.set_data(frq, abs(Y))

    Y = np.fft.fft(in_signal)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    fft2.set_data(frq, abs(Y))

    #filter
    filtered_sig = signal.lfilter(b, a, mixed_sig)
    filtered.set_data(samples, filtered_sig)

    #show output of accumulated absolute filtered signal against frequency
    amp = np.sum(np.abs(filtered_sig))
    if sweep_freq == 0:
            filtered_time.set_xdata([])
            filtered_time.set_ydata([])
    filtered_time.set_xdata(np.append(filtered_time.get_xdata(), sweep_freq))
    filtered_time.set_ydata(np.append(filtered_time.get_ydata(), amp))

    return sig_LO, sig_input, mixed, fft1, fft2, fftmixed, filtered

def setup_plots():
    # do all the plots

    # plot input signal
    fig = plt.figure()
    ax1 = fig.add_subplot(3,2,1)
    ax1.set_title("input signals and LO %d->%d Hz" % (start_freq, stop_freq))
    ax1.set_xlabel('time')
    ax1.set_ylabel('amp')
    ax1.set_xlim(0, time)
    ax1.set_ylim(-1, 1)
    ax1.grid()

    sig_LO, = ax1.plot([],[], 'r', linewidth=LW)
    sig_input, = ax1.plot([],[], 'y', linewidth=LW)

    ax2 = fig.add_subplot(3,2,2)
    ax2.set_title("mixed = inputs * LO")
    ax2.set_xlabel('time')
    ax2.set_ylabel('amp')
    ax2.set_xlim(0, time)
    ax2.set_ylim(-1, 1)
    ax2.grid()
    mixed, = ax2.plot([], [], 'b', linewidth=LW)


    ax3 = fig.add_subplot(3,2,4)
    ax3.set_title("fft of mixed")
    ax3.set_xlabel('Freq (Hz)')
    ax3.set_ylabel('|Y(freq)|')
    ax3.set_xlim(0, fs/2)
    ax3.set_ylim(0, 0.10)
    ax3.grid()
    fftmixed, = ax3.plot([], [], 'b', linewidth=LW)

    ax4 = fig.add_subplot(3,2,3)
    ax4.set_title("fft of inputs & LO")
    ax4.set_xlabel('Freq (Hz)')
    ax4.set_ylabel('|Y(freq)|')
    ax4.set_xlim(0, fs/2)
    ax4.set_ylim(0, 0.5)
    ax4.grid()
    fft1, = ax4.plot([], [], 'r', linewidth=LW)
    fft2, = ax4.plot([], [], 'y', linewidth=LW)

    ax5 = fig.add_subplot(3,2,5)
    ax5.set_title("filtered IF")
    ax5.grid()
    ax5.set_xlim(0, time)
    ax5.set_ylim(-0.1, 0.1)
    filtered, = ax5.plot([], [], 'b', linewidth=LW)

    ax6 = fig.add_subplot(3,2,6)
    ax6.set_title("filtered IF over time")
    ax6.grid()
    ax6.set_xlim(0, fs/2)
    ax6.set_ylim(0, 10)
    filtered_time, = ax6.plot([], [], 'b', linewidth=LW)

    plt.tight_layout()

    return fig,sig_LO, sig_input, mixed, fft1, fft2, fftmixed, filtered, filtered_time

def freq_range():
    return iter( range(start_freq, stop_freq, step_freq))

if __name__ == '__main__':
    fig, sig_LO, sig_input, mixed, fft1, fft2, fftmixed, filtered, filtered_time = setup_plots()
    line_ani = animation.FuncAnimation(fig, plots, freq_range, repeat=True, fargs=(sig_LO, sig_input, mixed, fft1, fft2, fftmixed, filtered, filtered_time), interval=50, blit=False)
    plt.show()
