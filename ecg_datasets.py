import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
from scipy.signal import spectrogram
import torch.nn.functional as F
import random

class RawECGDataset(Dataset):
    def __init__(self, filepaths, labels, seq_len=2500, augment=False, resize=False):
        self.filepaths = filepaths
        self.labels = labels
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        record = wfdb.rdrecord(self.filepaths[idx])
        signal = record.p_signal.T

        if signal.shape[1] < self.seq_len:
            pad_width = self.seq_len - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)))
        else:
            signal = signal[:, :self.seq_len]

        if self.augment:
            signal = self.augment_signal(signal)

        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label

    def augment_signal(self, signal):
        signal = signal * np.random.uniform(0.9, 1.1)
        signal = signal + np.random.normal(0, 0.01, size=signal.shape)
        signal = np.roll(signal, np.random.randint(-20, 20), axis=1)
        return signal


class SpectrogramECGDataset(Dataset):
    def __init__(self, filepaths, labels, seq_len=2500, fs=500, nperseg=128, noverlap=64, augment=False, resize=False):
        self.filepaths = filepaths
        self.labels = labels
        self.seq_len = seq_len
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.augment = augment
        self.resize = resize

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        record = wfdb.rdrecord(self.filepaths[idx])
        signal = record.p_signal.T  # (12, N)

        if signal.shape[1] < self.seq_len:
            pad_width = self.seq_len - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)))
        else:
            signal = signal[:, :self.seq_len]

        specs = []
        for channel in signal:
            f, t, Sxx = spectrogram(channel, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            Sxx = np.log(Sxx + 1e-10)
            specs.append(Sxx)

        spec_tensor = np.stack(specs, axis=0)  # (12, F, T)

        if self.augment:
            spec_tensor = self.augment_spectrogram(spec_tensor)

        spec_tensor = torch.tensor(spec_tensor, dtype=torch.float32)

        if self.resize:
            spec_tensor = spec_tensor.unsqueeze(0)
            spec_tensor = F.interpolate(spec_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            spec_tensor = spec_tensor.squeeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec_tensor, label

    def augment_spectrogram(self, spec):
        spec += np.random.normal(0, 0.01, size=spec.shape)

        for _ in range(random.randint(1, 2)):
            f_start = random.randint(0, spec.shape[1] // 2)
            f_width = random.randint(5, 15)
            spec[:, f_start:f_start + f_width, :] = 0

        for _ in range(random.randint(1, 2)):
            t_start = random.randint(0, spec.shape[2] // 2)
            t_width = random.randint(5, 15)
            spec[:, :, t_start:t_start + t_width] = 0

        return spec


def tokenize_biosignal(data):
    # data shape: (12, N)
    if data.ndim == 2 and data.shape[0] > 1:
        data = np.mean(data, axis=0)  # shape (N,)

    # Truncate, normalize, scale
    if data.shape[0] > 500:
        data = data[-500:]
    else:
        data = np.pad(data, (0, 500 - data.shape[0]))

    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
    data *= 100

    return np.round(data)


class TokenizedECGDataset(Dataset):
    def __init__(self, filepaths, labels, seq_len=500, augment=False, resize=False):
        self.filepaths = filepaths
        self.labels = labels
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        record = wfdb.rdrecord(self.filepaths[idx])
        signal = record.p_signal.T

        if self.augment:
            signal = self.augment_signal(signal)

        tokenized = tokenize_biosignal(signal)
        token_tensor = torch.tensor(tokenized, dtype=torch.long)

        return token_tensor, torch.tensor(self.labels[idx], dtype=torch.long)

    def augment_signal(self, signal):
        signal = signal + np.random.normal(0, 0.01, size=signal.shape)
        signal = signal * np.random.uniform(0.9, 1.1)
        return signal
