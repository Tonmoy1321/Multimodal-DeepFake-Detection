import os
import numpy as np
import librosa
import subprocess
from tqdm import tqdm

class AudioPreprocessor:
    def __init__(self, dataset_dir, output_dir, sr=16000, n_mfcc=13, train_frames=120, eval_frames=24, test_frames=24):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.sr = sr  
        self.n_mfcc = n_mfcc
        self.train_frames = train_frames
        self.test_frames = test_frames
        self.eval_frames = eval_frames

        # Define MFCC extraction parameters
        self.n_fft = int(0.025 * sr)  # 25ms window size
        self.hop_length = int(0.010 * sr)  # 10ms hop length

        # Create output directories for train, eval, test
        for split in ["train", "eval", "test"]:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    def extract_audio(self, video_path):
        """Extracts audio from video and converts it to MFCCs."""
        audio_path = video_path.replace(".mp4", ".wav")

        # Extract audio using ffmpeg
        try:
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-ar", str(self.sr), "-ac", "1", "-y", audio_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from {video_path}: {e}")
            return None

        # Load and process audio
        audio, _ = librosa.load(audio_path, sr=self.sr)

        # Compute MFCCs with 25ms window and 10ms hop length
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        return mfccs.T  # Shape: (time, n_mfcc)

    def process_videos(self):
        """Processes videos in subdirectories of fake and real directories, extracting only audio."""
        for label in ["fake", "real"]:
            label_dir = os.path.join(self.dataset_dir, label)

            for subfolder in tqdm(os.listdir(label_dir), desc=f"Processing {label} videos"):
                subfolder_path = os.path.join(label_dir, subfolder)

                if not os.path.isdir(subfolder_path):  # Skip if it's not a folder
                    continue

                for video_file in os.listdir(subfolder_path):
                    video_path = os.path.join(subfolder_path, video_file)

                    if not video_file.endswith(".mp4"):
                        continue  # Skip non-video files

                    # Extract audio features
                    mfcc_features = self.extract_audio(video_path)

                    if mfcc_features is not None:
                        # Assign audio frames to train, eval, and test
                        total_frames = len(mfcc_features)
                        if total_frames >= (self.train_frames + self.eval_frames + self.test_frames):
                            train_data = mfcc_features[:self.train_frames]
                            eval_data = mfcc_features[self.train_frames:self.train_frames + self.eval_frames]
                            test_data = mfcc_features[self.train_frames + self.eval_frames:self.train_frames + self.eval_frames + self.test_frames]

                            # Save with appropriate labels
                            prefix = f"{label}_"
                            file_name = f"{subfolder}_{video_file.replace('.mp4', '.npy')}"

                            np.save(os.path.join(self.output_dir, "train", f"{prefix}{file_name}"), np.array(train_data))
                            np.save(os.path.join(self.output_dir, "eval", f"{prefix}{file_name}"), np.array(eval_data))
                            np.save(os.path.join(self.output_dir, "test", f"{prefix}{file_name}"), np.array(test_data))

                            print(f"Processed {video_file} in {subfolder}: Train={len(train_data)}, Eval={len(eval_data)}, Test={len(test_data)}")

                        else:
                            print(f"Skipping {video_file} in {subfolder}: Not enough valid audio frames (Found {total_frames}).")

if __name__ == "__main__":
    dataset_directory = "./fakeavceleb"
    output_directory = "./processed_audio"
    preprocessor = AudioPreprocessor(dataset_directory, output_directory)
    preprocessor.process_videos()