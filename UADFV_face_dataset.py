import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

class VideoPreprocessor:
    def __init__(self, dataset_dir, output_dir, frame_size=(256, 256), train_frames=100, eval_frames=20, test_frames=20):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.frame_size = frame_size
        self.train_frames = train_frames
        self.eval_frames = eval_frames
        self.test_frames = test_frames

        # Create output directories for train, eval, test
        for split in ["train", "eval", "test"]:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)

        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(image_size=frame_size[0], margin=20, keep_all=False, post_process=False)

    def extract_faces(self, video_path):
        """Reads a video, detects faces in full-size frames, crops the face, resizes it, and keeps only frames with detected faces."""
        cap = cv2.VideoCapture(video_path)
        valid_faces = []
        success, frame = cap.read()

        while success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR
            boxes, _ = self.mtcnn.detect(frame_rgb)  # Detect face

            if boxes is not None:  # If a face is detected
                x1, y1, x2, y2 = map(int, boxes[0])  # Get the first detected face box
                face_crop = frame_rgb[y1:y2, x1:x2]  # Crop the detected face

                if face_crop.size > 0:  # Ensure valid cropping
                    face_resized = cv2.resize(face_crop, self.frame_size)  # Resize to desired frame size
                    valid_faces.append(face_resized)

            success, frame = cap.read()

        cap.release()
        return valid_faces

    def process_videos(self):
        """Processes videos directly in fake and real folders, saving them into train, eval, and test sets."""
        for label in ["fake", "real"]:
            label_dir = os.path.join(self.dataset_dir, label)

            for video_file in tqdm(os.listdir(label_dir), desc=f"Processing {label} videos"):
                if not video_file.endswith(".mp4"):
                    continue  # Skip non-video files

                video_path = os.path.join(label_dir, video_file)
                video_id = video_file.split(".mp4")[0]  # Extract video_id from filename

                faces = self.extract_faces(video_path)

                if len(faces) >= (self.train_frames + self.eval_frames + self.test_frames):
                    # Assign frames to train, eval, and test
                    train_data = faces[:self.train_frames]
                    eval_data = faces[self.train_frames:self.train_frames + self.eval_frames]
                    test_data = faces[self.train_frames + self.eval_frames:self.train_frames + self.eval_frames + self.test_frames]

                    # Save with appropriate labels
                    prefix = f"{label}_"
                    np.save(os.path.join(self.output_dir, "train", f"{prefix}{video_id}.npy"), np.array(train_data))
                    np.save(os.path.join(self.output_dir, "eval", f"{prefix}{video_id}.npy"), np.array(eval_data))
                    np.save(os.path.join(self.output_dir, "test", f"{prefix}{video_id}.npy"), np.array(test_data))
                else:
                    print(f"Not enough valid faces detected in video: {video_path} (Found {len(faces)})")

if __name__ == "__main__":
    dataset_directory = "./UADFV"  
    output_directory = "./processed"
    
    preprocessor = VideoPreprocessor(dataset_directory, output_directory)
    preprocessor.process_videos()
