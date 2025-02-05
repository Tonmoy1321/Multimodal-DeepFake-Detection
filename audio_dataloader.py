import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.npy_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_file = self.npy_files[idx]
        mfcc_data = np.load(npy_file)  # Shape: (120, 13)

        filename = os.path.basename(npy_file)
        label = 0 if filename.split("_")[0].lower() == "real" else 1

        mfcc_data = torch.tensor(mfcc_data, dtype=torch.float32).unsqueeze(1)  # Shape: (120, 1, 13)
        mfcc_data = mfcc_data.repeat(1, 3, 1)  # Shape: (120, 3, 13)

        return mfcc_data, torch.tensor([label], dtype=torch.float32)

def get_audio_dataloader(folder_path, batch_size=1, shuffle=False):
    dataset = AudioDataset(folder_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def collate_fn(batch):
    mfccs, labels = zip(*batch)
    max_seq_len = max([m.size(0) for m in mfccs])  # Ensure all sequences are padded to the same length
    batch_size = len(mfccs)

    padded_batch = torch.zeros((batch_size, max_seq_len, 3, 13), dtype=torch.float32)

    for i, m in enumerate(mfccs):
        seq_len = m.size(0)
        padded_batch[i, :seq_len, :, :] = m  # Pad sequences to max_seq_len

    labels = torch.stack(labels)

    return padded_batch, labels