import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FaceDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): Path to the folder containing .npy files.
        """
        self.folder_path = folder_path
        self.npy_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        """
        Loads a single .npy file and extracts the ground truth label from the filename.
        """
        npy_file = self.npy_files[idx]
        face_data = np.load(npy_file)  # Shape: (num_frames, 256, 256, 3)

        # Extract label from filename (real_ -> 0, fake_ -> 1)
        filename = os.path.basename(npy_file)  # Get filename without path
        label_str = filename.split("_")[0]  # Extract 'real' or 'fake'
        label = 0 if label_str.lower() == "real" else 1  # Convert to numeric label

        # Convert to tensor and normalize to [0, 1]
        face_data = torch.tensor(face_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # Shape: (num_frames, 3, 256, 256)

        return face_data, torch.tensor([label], dtype=torch.float32)  # Return face tensor & label
    

def get_face_dataloader(folder_path, batch_size=1, shuffle=False):
    """
    Args:
        folder_path (str): Path to the folder containing .npy files.
        batch_size (int): Number of videos to load per batch (Default is 1 for loading one at a time).
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Torch DataLoader for the dataset.
    """
    dataset = FaceDataset(folder_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, collate_fn=collate_fn)

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences across files.
    Pads sequences to the length of the longest sequence in the batch.
    """
    videos, labels = zip(*batch) 
    max_seq_len = max([v.size(0) for v in videos])
    batch_size = len(videos)
    padded_batch = torch.zeros((batch_size, max_seq_len, 3, 256, 256), dtype=torch.float32)

    for i, v in enumerate(videos):
        padded_batch[i, :v.size(0)] = v  # Copy frames into the batch

    labels = torch.stack(labels)  # Convert labels into a batch tensor

    return padded_batch, labels