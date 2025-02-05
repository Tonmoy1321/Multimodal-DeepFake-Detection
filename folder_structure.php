project/
│
├── script.py            # The script you're asking me to modify
│
├── dataset/             # Dataset folder
│   ├── fake/            # Contains folders for deepfake videos
│   │   ├── video_id_1/  # Folder named after video id
│   │   │   └── video.mp4  # Actual video file
│   │   ├── video_id_2/
│   │   │   └── video.mp4
│   │   └── ...          # More folders for other deepfake videos
│   │
│   ├── real/            # Contains folders for real videos
│   │   ├── video_id_1/  # Folder named after video id
│   │   │   └── video.mp4  # Actual video file
│   │   ├── video_id_2/
│   │   │   └── video.mp4
│   │   └── ...          # More folders for other real videos
│
└── processed(will be created by the script)/           # Directory for processed outputs 
    ├── fake/......................                     # Processed files for fake videos
    └── real/ .....................                     # Processed files for real videos
