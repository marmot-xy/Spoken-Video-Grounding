import numpy as np
import torch
from gensim.utils import tokenize
from torch.utils.data import Dataset

from utils.accessor import load_json
from utils.container import resourceContainer
from utils.helper import sequence_mask


class BaseDataset(Dataset):
    def __init__(self, data_path, vocab_path, max_frame_num, max_word_num, **kwargs):
        self.vocab = resourceContainer.fetch_vocab(vocab_path)
        self.data = load_json(data_path)
        self.ori_data = self.data
        self.max_frame_num = max_frame_num
        self.max_word_num = max_word_num

    def load_data(self, data):
        self.data = data

    def _load_frame_features(self, vid):
        raise NotImplementedError

    def _load_audio_features(self, aid):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid, duration, timestamps, sentence, aid = self.data[index]
        duration = float(duration)

        words = [w.lower() for w in tokenize(sentence)]
        words = [w for w in words if w in self.vocab]
        visual_feat = self._load_frame_features(vid)
        audio_feat, audio_frame = self._load_audio_features(aid)
        textual_feat = [self.vocab[w].astype(np.float32) for w in words]

        if not textual_feat:
            return self.__getitem__(index + 1)

        return {
            'visual_feat': visual_feat,
            'textual_feat': textual_feat,
            'audio_feat': audio_feat,
            'audio_frame': audio_frame,
            'raw': [vid, duration, timestamps, sentence, aid]
        }


def build_collate_data(max_frame_num, max_word_num, max_audio_num, frame_dim, word_dim, audio_dim):
    def collate_data(samples):
        bsz = len(samples)
        batch = {'raw': [sample['raw'] for sample in samples]}

        frames_len = []
        words_len = []
        audio_len = []

        for i, sample in enumerate(samples):
            frames_len.append(min(len(sample['visual_feat']), max_frame_num))
            words_len.append(min(len(sample['textual_feat']), max_word_num))
            audio_len.append(min(sample['audio_frame'], max_audio_num))

        visual_feat = np.zeros([bsz, max_frame_num, frame_dim]).astype(np.float32)
        textual_feat = np.zeros([bsz, max_word_num, word_dim]).astype(np.float32)
        audio_feat = np.zeros([bsz, max_audio_num, audio_dim]).astype(np.float32)
        start_idx = np.zeros([bsz]).astype(np.float32)
        end_idx = np.zeros([bsz]).astype(np.float32)

        for i, sample in enumerate(samples):
            word_len = min(len(sample['textual_feat']), max_word_num)
            textual_feat[i, :word_len] = sample['textual_feat'][:word_len]
            #audio_feat[i, ] = sample['audio_feat']

            start_idx[i] = max(0, sample['raw'][2][0] / sample['raw'][1] * max_frame_num)
            end_idx[i] = min(max_frame_num, sample['raw'][2][1] / sample['raw'][1] * max_frame_num)

            frame_idx = np.arange(0, max_frame_num + 1) / max_frame_num * len(sample['visual_feat'])
            frame_idx = np.round(frame_idx).astype(np.int64)
            frame_idx[frame_idx >= len(sample['visual_feat'])] = len(sample['visual_feat']) - 1
            frames_len[i] = visual_feat.shape[1]
            for j in range(visual_feat.shape[1]):
                s, e = frame_idx[j], frame_idx[j + 1]
                assert s <= e
                if s == e:
                    visual_feat[i, j] = sample['visual_feat'][s]
                else:
                    visual_feat[i, j] = sample['visual_feat'][s:e].mean(axis=0)

            audio_idx = np.arange(0, max_audio_num + 1) / max_audio_num * len(sample['audio_feat'])
            audio_idx = np.round(audio_idx).astype(np.int64)
            audio_idx[audio_idx >= len(sample['audio_feat'])] = len(sample['audio_feat']) - 1
            audio_len[i] = audio_feat.shape[1]
            for j in range(audio_feat.shape[1]):
                s, e = audio_idx[j], audio_idx[j + 1]
                assert s <= e
                if s == e:
                    audio_feat[i, j] = sample['audio_feat'][s]
                else:
                    audio_feat[i, j] = sample['audio_feat'][s:e].mean(axis=0)


        batch.update({
            'net_input': {
                'visual_feat': torch.from_numpy(visual_feat),
                'textual_feat': torch.from_numpy(textual_feat),
                'audio_feat': torch.from_numpy(audio_feat),
                'textual_mask': sequence_mask(torch.from_numpy(np.asarray(words_len)), max_word_num),
                'audio_mask': sequence_mask(torch.from_numpy(np.asarray(audio_len)), max_audio_num),
                'audio_length': audio_len,
                'start': torch.tensor(start_idx),
                'end': torch.tensor(end_idx)
            },
            'target': {
                'start': torch.tensor(start_idx),
                'end': torch.tensor(end_idx)
            }
        })

        return batch

    return collate_data
