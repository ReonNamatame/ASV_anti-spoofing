# GMMs(Gaussian Mixture Models) front-end are LFCCs and CQCCs.
from baseline.lfcc import *
import librosa
import glob
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

# speaker embedding by using GMMs, where n_components = 512
genuine_gmms = GaussianMixture(n_components=512, covariance_type='diag', max_iter=100, random_state=None)

spoofed_gmms = GaussianMixture(n_components=512, covariance_type='diag', max_iter=100, random_state=None)

# Train the other parameters using the EM algorithm
genuine_gmms.fit()

spoofed_gmms.fit()

speech_count = 0
total_speech = len(glob.glob('the path to the database(ASVspoof2019) LA development'))

for speech in sorted(glob.glob('the path to the database')):
    
    print(float(speech_count/total_speech),'percent completed')
    
    lfcc = LFCC(wavfile=speech).get_lfcc()

    score = np.array([])
    for i, lfcc_frame in enumerate(lfcc.T):
        loglh_genuine = genuine_gmms.score(lfcc_frame.reshape(1, -1))
        loglh_spoofed = spoofed_gmms.score(lfcc_frame.reshape(1, -1))
    
    # compute mean

    # compute log-likelihood ratio
    score = loglh_genuine - loglh_spoofed

    speech_count = speech_count + 1