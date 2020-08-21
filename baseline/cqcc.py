
import wave
import numpy as np
import scipy.signal
from scipy import arange, complex128, exp, hamming, log2, zeros
from scipy import pi as mpi
from scipy.io.wavfile import read
# CQCCs(Constant Q-Cepstrum Coefficients)
# 1. CQT(Constant Q-Transform) computing from waveform
# 2. Power spectrum from CQT
# 3. Take a log
# 4. Uniform resampling
# 5. DCT(Discrete Cosine Transform)
# #
# beginning of class CQCC
class CQCC(object):
    def __init__(self, wave, sr=44100, B=96, fmax=sr//2, fmin=0, d=16, cf=19):
        # input parameters
        # wave  : input signal
        # fs    : sampling frequency
        # B     : number of bins per octave [default = 96]
        # fmax  : highest frequency to be analyzed [default = Nyquist frequency]
        # fmin  : lowest frequency to be analyzed [default = ~20Hz to fullfill an integer number of octave]
        # d     : number of uniform samples in the first octave [default 16]
        # cf    : number of cepstral coefficients excluding 0'th coefficient [default 19]
        # 
        # output parameters
        # CQCCs         : constant Q-cepstral coefficients (nCoeff x nFea)
        # LogP_absCQT   : log power magnitude spectrum of constant Q transform
        # TimeVec       : time at the centre of each frame [sec]
        # FreqVec       : center frequencies of analysis filters [Hz]
        # Ures_LogP_absCQT  : uniform resampling of LogP_absCQT
        # Ures_FreqVec  : uniform resampling of FreqVec [Hz]
        # #
        self.wave = wave
        self.sr = sr
        self.B = B
        self.fmax = fmax
        self.fmin = fmin
        self.d = d
        self.cf = cf

    def get_cqcc(self, cf):
        
        gamma = 228.7*(2^(1/B)-2^(-1/B))
        # CQT computing
        Xcq = _cqt(x, B, fs, fmin, fmax, 'rasterize', 'full', 'gamma', gamma)
        power_Xcq = np.abs(Xcq)**2

        # Log PowerSpectrum
        LogP_Xcq = np.log10(power_Xcq + eps)
        TimeVec = (1:size(absCQT,2))*Xcq.xlen/size(absCQT,2)/fs
        FreqVec = fmin*(2.^((0:size(absCQT,1)-1)/B))

        # Uniform Resampling
        kl = (B*log2(1+1/d))
        [Ures_LogP_absCQT, Ures_FreqVec] = resample(LogP_absCQT, FreqVec,1/(fmin*(2^(kl/B)-1)),1,1,'spline')
        
        # CQCC
        cqccs = _dct(Ures_LogP_absCQT)[:cf]

        return cqccs
    
    def _cqt(self):
        pass

    def _uniform_resampling(self):
        pass

    def _dct(self):
        pass

# end of class CQCC