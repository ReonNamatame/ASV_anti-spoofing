#import wave
import numpy as np
import scipy.signal
from scipy.io.wavfile import read
from scipy.fftpack import realtransforms

import matplotlib.pyplot as plt

# Linear Frequency Cepstral Coefficients(LFCCs)
#beginning of class LFCC
class LFCC(object):

#public method
    def __init__(self, wavfile="record.wav"):
        #%time wave.open(str(wavfile), mode="rb")
        #%time read(wavfile)
        self._sr, self._waveform = read(wavfile)
        
    def get_audio_signal(self):
        return self._waveform
    
    def get_sampling_rate(self):
        return self._sr
    
    def get_lfcc(self, p=0.97, nfft=512, nchannels=24, nceps=12):
        
        # define a convolute preEmphasis filter
        self._waveform = self._preEmphasis(self._waveform, p)
        #print(type(self._waveform))
        
        # define a hamming window
        self._nfft = nfft
        hammingWindow = np.hamming(self._nfft)
        
        self._hop_length = self._nfft//2
        # make a spectrogram
        spec = self._stft(wave=self._waveform, window=hammingWindow, step=self._hop_length)
        # n: fft_bin, d: cycle time
        freq = np.fft.fftfreq(n=self._nfft//2, d=1.0/self._sr)

        """
        for i, sp in enumerate(spec):
            plt.plot(freq, sp)
        plt.show()
        """
        """
        import scipy.io.wavfile as wav
        import scipy.signal as signal
        import librosa

        f, t, Zxx = signal.stft(self._waveform, fs=self._sr)
        #plt.pcolormesh(t, f, , shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.show()

        #DB = librosa.amplitude_to_db(Zxx, ref=np.max)
        #lirosa.display.specshow(DB, sr=self._sr, hop_length=self._nfft, x_axis='time', y_axis='log')
        #plt.colorbar(format='%+2.0fdB')
        """

        #linearfilterbank
        df = self._sr / self._nfft # frequency resolution
        print("sampling rate: {}, freq resolution: {}".format(self._sr, df))
        filterbank = self._linearFilterBank(nchannels=nchannels)
        print(filterbank.shape)

        """
        for c in np.arange(0, nchannels):
            plt.plot(np.arange(0, self._nfft//2) * df, filterbank[c])
        plt.show()
        """
        # apply linearfilterbanks for each vector, then sum all and take log
        linear_spec = np.log10(np.dot(spec, filterbank.T))
        print("linear_spec:", linear_spec.shape)
        
        # obtain a cepstrum by applying discrete cosine transform to log-linear spectrum
        cepstrum = self._dct(linear_spec).T
        
        # cepstrum = (n-dimensional feature, shift), nceps is the number of features to use
        lfccs = cepstrum[:nceps]

        return lfccs

    #pre-emphasis filter
    def _preEmphasis(self, signal, p):
        #signal := voice_signal, p := coefficient
        #make FIR filter such that coefficients are (1.0, p)
        return scipy.signal.lfilter([1.0, -p], 1.0, signal)
    
    #convert hz to mel
    def _hz2mel(self, f):
        return 1125.0 * np.log(f/700.0 + 1.0)
    
    #convert mel to hz
    def _mel2hz(self, m):
        return 700.0 * (np.exp(m/1125.0) - 1.0)
    
    #Short Time Fourier Transform: STFT
    def _stft(self, wave, window, step):
        # wave: waveform (numpy.ndarray)
        # window: hammingWindow (numpy.ndarray)
        # step: overlapping length
        wavelen = wave.shape[0]
        windowlen = window.shape[0] # windowLength, fft_bin, cripping_size
        shift = (wavelen - windowlen + step-1) // step + 1
        print("wavelen: {}, fft_bin: {}, shift: {}".format(wavelen, windowlen, shift))

        #padded_wave = np.zeros(wavelen+step)
        #padded_wave = np.zeros(windowlen+(shift*step))
        #padded_wave[:wavelen] = wave # waveform has to be reformed to fit the sliding window 
        
        X = np.array([]) # X: spectrum will have to be (shift, windowlen)
        for m in range(shift):
            start = step * m
            x = np.fft.fft(wave[start:start+windowlen]*window, norm='ortho')[:self._nfft//2]
            if m == 0:
                X = np.hstack((X, x))
            else:
                X = np.vstack((X, x))
            """
            if m == 120:
                plt.plot(np.arange(0, windowlen), wave[start:start+windowlen]*window)
                plt.show()
                plt.plot(np.fft.fftfreq(n=self._nfft, d=1.0/self._sr)[:self._nfft//2], np.abs(x)**2)
                plt.show()
            """
        print(X.shape)
        # return power-spectrum
        return np.abs(X)**2/self._nfft
    
    #generate linearfilterbank
    def _linearFilterBank(self, nchannels=40):
        freq_min = 0
        freq_max = self._sr//2
        linear_centers = np.linspace(freq_min, freq_max, nchannels+2)
        bins = np.floor((self._nfft+1)*linear_centers/self._sr)
        #print(bins)

        filterbank = np.zeros((nchannels, self._nfft//2))
        for m in range(1, nchannels+1):
            f_m_minus = int(bins[m - 1])    # left
            f_m = int(bins[m])              # center
            f_m_plus = int(bins[m + 1])     # right

            for k in range(f_m_minus, f_m):
                filterbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
            for k in range(f_m, f_m_plus):
                filterbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

        return filterbank
    
    #discrete cosine transform
    def _dct(self, mspec):
        return realtransforms.dct(mspec, type=2, norm='ortho', axis=-1)
    
#end of class LFCC

if __name__ == "__main__":
    #import librosa

    lfcc = LFCC("utterance3.wav").get_lfcc().T
    
#    y, sr = librosa.load("utterance3.wav", sr=22050)
#    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12).T

    print(lfcc[100], lfcc.shape)
#   print(mfcc[100], mfcc.shape)