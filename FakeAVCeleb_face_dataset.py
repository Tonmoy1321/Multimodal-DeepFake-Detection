import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from mtcnn import MTCNN


class VideoPreprocessor:
    def __init__(self, dataset_dir, output_dir, frame_size=(256, 256)):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.frame_size = frame_size

        for split in ["train", "eval", "test"]:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)

        self.mtcnn = MTCNN()  # Initialize MTCNN from the mtcnn library

    def extract_frames(self, video_path, max_frames=300):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break  # Stop if we can't read more frames

            if frame_count >= max_frames:
                break  # Stop after collecting max_frames

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.frame_size)
            frames.append(frame_resized.astype(np.uint8))  # Save as uint8
            frame_count += 1

        cap.release()
        return frames

    def process_videos(self):
        for label in ["fake", "real"]:
            label_dir = os.path.join(self.dataset_dir, label)
            subfolders = [f for f in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, f))]

            random.shuffle(subfolders)
            train_split = int(len(subfolders) * 0.8)
            eval_split = int(len(subfolders) * 0.9)

            split_mapping = {
                "train": subfolders[:train_split],
                "eval": subfolders[train_split:eval_split],
                "test": subfolders[eval_split:]
            }

            for split, subfolder_list in split_mapping.items():
                for subfolder in tqdm(subfolder_list, desc=f"Processing {label} videos [{split}]"):
                    subfolder_path = os.path.join(label_dir, subfolder)

                    for video_file in os.listdir(subfolder_path):
                        video_path = os.path.join(subfolder_path, video_file)

                        if not video_file.endswith(".mp4"):
                            continue

                        frames = self.extract_frames(video_path)

                        if len(frames) > 0:
                            prefix = f"{label}_"
                            file_name = f"{subfolder}_{video_file.replace('.mp4', '.npy')}"
                            np.save(os.path.join(self.output_dir, split, f"{prefix}{file_name}"), np.array(frames))
                            print(f"Processed {video_file} in {subfolder} -> {split} (Frames: {len(frames)})")
                        else:
                            print(f"Skipping {video_file} in {subfolder}: No frames extracted.")

if __name__ == "__main__":
    dataset_directory = "./fakeavceleb"
    output_directory = "./processed_FAVC"
    preprocessor = VideoPreprocessor(dataset_directory, output_directory)
    preprocessor.process_videos()