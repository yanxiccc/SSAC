#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class AudioCaptionDataset(Dataset):

    def __init__(self, dataset='AudioCaps', split='train', task='captioning', return_dict=False):
        """
        load audio clip's waveform and corresponding caption
        Args:
            dataset: 'AudioCaps', 'Clotho
            split: 'train', 'val', 'test'
        """
        super(AudioCaptionDataset, self).__init__()
        self.dataset = dataset
        self.split = split
        self.task = task
        self.return_dict = return_dict
        self.h5_path = f'data/{dataset}/hdf5s/{split}/{split}.h5'
        if dataset == 'AudioCaps' and split == 'train':
            self.is_train = True
            self.num_captions_per_audio = 1
            with h5py.File(self.h5_path, 'r') as hf:
                self.audio_keys = [audio_name for audio_name in hf['audio_name'][:]]
                # audio_names: [str]
                self.captions = [caption for caption in hf['caption'][:]]
        else:
            self.is_train = False
            self.num_captions_per_audio = 5
            with h5py.File(self.h5_path, 'r') as hf:
                self.audio_keys = [audio_name for audio_name in hf['audio_name'][:]]
                self.captions = [caption for caption in hf['caption'][:]]
                if dataset == 'Clotho':
                    self.audio_lengths = [length for length in hf['audio_length'][:]]
                # [cap_1, cap_2, ..., cap_5]

    def __len__(self):
        if self.task == 'captioning' and self.split != 'train' and self.return_dict:
            return len(self.audio_keys)
        else:
            return len(self.audio_keys) * self.num_captions_per_audio

    def __getitem__(self, index):

        if self.task == 'captioning' and self.split != 'train' and self.return_dict:
            audio_idx = index
        else:
            audio_idx = index // self.num_captions_per_audio
        audio_name = self.audio_keys[audio_idx]
        with h5py.File(self.h5_path, 'r') as hf:
            waveform = hf['waveform'][audio_idx]

        if self.dataset == 'AudioCaps' and self.is_train:
            caption = self.captions[audio_idx]
        else:
            captions = self.captions[audio_idx]
            if self.task == 'captioning' and self.split != 'train' and self.return_dict:
                caption_field = ['caption_{}'.format(i) for i in range(1, self.num_captions_per_audio + 1)]
                caption = {}
                for i, cap_ind in enumerate(caption_field):
                    caption[cap_ind] = captions[i].decode()
            # elif self.task == 'captioning' and self.split != 'train' and self.return_dict is not True:
            #     cap_idx = index % self.num_captions_per_audio
            #     caption = captions[cap_idx].decode()
            else:
                cap_idx = index % self.num_captions_per_audio
                caption = captions[cap_idx]

        if self.dataset == 'Clotho':
            length = self.audio_lengths[audio_idx]
            return waveform, caption, audio_idx, length, index, audio_name
        else:
            return waveform, caption, audio_idx, len(waveform), index, audio_name


def collate_fn(batch_data):
    """

    Args:
        batch_data:

    Returns:

    """

    max_audio_length = max([i[3] for i in batch_data])

    wav_tensor = []
    for waveform, _, _, _, _, _ in batch_data:
        if max_audio_length > waveform.shape[0]:
            padding = torch.zeros(max_audio_length - waveform.shape[0]).float()
            temp_audio = torch.cat([torch.from_numpy(waveform).float(), padding])
        else:
            temp_audio = torch.from_numpy(waveform[:max_audio_length]).float()
        wav_tensor.append(temp_audio.unsqueeze_(0))

    wavs_tensor = torch.cat(wav_tensor)
    captions = [i[1] for i in batch_data]
    audio_ids = torch.Tensor([i[2] for i in batch_data])
    indexs = np.array([i[4] for i in batch_data])
    audio_names = [i[5] for i in batch_data]

    return wavs_tensor, captions, audio_ids, indexs, audio_names


def get_dataloader(split, config, return_dict=False):
    dataset = AudioCaptionDataset(config.dataset, split, 'captioning', return_dict)
    if split == 'train':
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False

    return DataLoader(dataset=dataset,
                      batch_size=config.data.batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      num_workers=config.data.num_workers,
                      collate_fn=collate_fn)


