import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# signal -> amp -> mixer -> IF -> bp filter -> detector -> lpf -> out
#                   |
#         sweep ->  LO

# output as a plot of frequencies vs amplitudes


# sweep parameters
start_freq= 50.0 # Hz
step_freq = 100

# sampling rate
fs = float(10000)
end_freq = fs / 2
freq_bands = range(int(start_freq), int(end_freq), step_freq)
print("sampling rate %d Hz" % fs)

# IF filter params
IF_bw = 50
IF_order = 3


# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y



# detector parameters
detect_samps = 400
print("detector samples = %d" % detect_samps)

# LPF parameters - just take average for now


# generate sweep
sweep_samples = (((end_freq - start_freq) / step_freq) * detect_samps)
sweep_time = sweep_samples / fs
samples = np.linspace(0, sweep_time, int(fs*sweep_time), endpoint=False)
print("sweep samples = %d (%f s), from %d to %d Hz" % (sweep_samples, sweep_time, start_freq, end_freq))
print("%d freq bands" % len(freq_bands))
sweep = signal.chirp(samples, start_freq, sweep_time, end_freq)

# signal params - use a sine for now
input_freq = 1500 * 2 # Hz
print("input signal freq %d Hz" % input_freq)
in_signal = np.sin(2 * np.pi * input_freq * samples)
#in_signal = signal.square(2 * np.pi * input_freq * samples)

def generate_IF_filters():
    IFs = []
    # for each frequency, generate a bp
    for freq in freq_bands:
        b, a = butter_bandpass(freq-IF_bw/2, freq+IF_bw/2, fs, order=IF_order)
        w, h = signal.freqz(b, a)
        IFs.append((w,h))
    return IFs

def plots():
    # do all the plots
    num_plots = 6

    # plot input signal
    fig = plt.figure()
    ax = fig.add_subplot(num_plots,1,1)
    ax.set_title("input signal at %d Hz" % input_freq)
    ax.set_xlabel('time')
    ax.set_ylabel('amp')
    ax.plot(samples, in_signal)

    # plot sweep signal
    ax = fig.add_subplot(num_plots,1,2)
    ax.set_title("sweep from %d to %d Hz" % (start_freq, end_freq))
    ax.set_xlabel('time')
    ax.set_ylabel('amp')
    ax.plot(samples, sweep)

    # plot mixed signal
    mixed = sweep * in_signal

    ax = fig.add_subplot(num_plots,1,3)
    ax.set_title("IF (input * sweep)")
    ax.set_xlabel('time')
    ax.set_ylabel('amp')
    ax.plot(samples, mixed)

    # bandpass
    ax = fig.add_subplot(num_plots,1,4)
    ax.set_title('%d band pass filters' % len(freq_bands))
    ax.set_xlabel('Frequency hz')
    ax.set_ylabel('[dB]')
    IFs = generate_IF_filters()
    for IF in IFs:
        w, h = IF
        ax.plot((fs * 0.5 / np.pi) * w, abs(h))

    ax = fig.add_subplot(num_plots,1,5)
    ax.set_title("filtered IF in each band")
    ax.set_xlabel('time')
    ax.set_ylabel('amp')
    # then for each bandpass, feed it that section of the signal that matches with the sweep
    filtered_signal = []
    accumulators = []
    bands = []
    start_samp = 0
    for freq in freq_bands:
        data = mixed[start_samp:start_samp+detect_samps]
        start_samp += detect_samps
        filtered = butter_bandpass_filter(data, freq-IF_bw/2, freq+IF_bw/2, fs, order=IF_order)
        filtered_signal.extend(filtered)
        bands.append(freq)
        accumulators.append(np.sum(np.abs(filtered)))
    
    ax.plot(samples, filtered_signal)

    ax = fig.add_subplot(num_plots,1,6)
    ax.set_title("summed absolute filtered IF in each band")
    ax.set_xlabel('freq')
    ax.set_ylabel('amp')
    # then for each bandpass, feed it that section of the signal that matches with the sweep
    ax.plot(bands, accumulators)

    # show it
    #plt.tight_layout()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plots()
