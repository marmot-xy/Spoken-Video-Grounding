import os
import scipy
import h5py
import numpy as np
import librosa
from dataset.moment_retrieval.base import BaseDataset, build_collate_data



def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

class AnetCaption(BaseDataset):
    def __init__(self, data_path_template, vocab_path, max_frame_num, max_word_num, frame_dim, word_dim,
                 feature_path, audio_path, action, max_audio_num,audio_dim, **kwargs):
        super().__init__(data_path_template.format(action), vocab_path, max_frame_num, max_word_num)
        self.feature_path = feature_path
        self.audio_path = audio_path
        self.collate_fn = build_collate_data(max_frame_num, max_word_num, max_audio_num, frame_dim, word_dim, audio_dim)
        self.video_pool = {}



    def _load_frame_features(self, vid):
        if vid in self.video_pool:
            return self.video_pool[vid]
        else:
            with h5py.File(os.path.join(self.feature_path, '%s.h5' % vid), 'r') as fr:
                features = np.asarray(fr['feature']).astype(np.float32)
            self.video_pool[vid] = features
            return features

    def _load_audio_features(self, aid):
        audio_path = os.path.join(self.audio_path, '%s' % aid)
        audio_feats, n_frames = self.LoadAudio(audio_path)
        return audio_feats, n_frames

    def LoadAudio(self, path, target_length=800, use_raw_length=False):
        """ Step 3
            Convert audio wav file to mel spec feats
            target_length is the maximum number of frames stored (disable with use_raw_length)
            # NOTE: assumes audio in 16 kHz wav file
        """
        audio_type = 'melspectrogram'
        preemph_coef = 0.97
        sample_rate = 16000
        window_size = 0.025
        window_stride = 0.01
        window_type = 'hamming'
        num_mel_bins = 128
        padval = 0
        fmin = 20
        n_fft = int(sample_rate * window_size)
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)
        windows = {'hamming': scipy.signal.hamming}
        # load audio, subtract DC, preemphasis
        # NOTE: sr=None to avoid resampling (assuming audio already at 16 kHz sr
        y, sr = librosa.load(path, sr=None)
        if y.size == 0:
            y = np.zeros(200)
        y = y - y.mean()
        y = preemphasis(y, preemph_coef)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length,
                            window=windows[window_type])
        spec = np.abs(stft) ** 2
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            feats = librosa.power_to_db(melspec, ref=np.max)
        n_frames = feats.shape[1]
        if use_raw_length:
            target_length = n_frames
        p = target_length - n_frames
        if p > 0:
            feats = np.pad(feats, ((0, 0), (0, p)), 'constant',
                           constant_values=(padval, padval))
        elif p < 0:
            feats = feats[:, 0:p]
            n_frames = target_length
        return feats.T, n_frames


    def collate_data(self, samples):
        return self.collate_fn(samples)
