from src.utils.video_processing import VideoProcessor

if __name__ == "__main__":
    processor = VideoProcessor(
        video_path="data/raw/video.mp4",
        output_dir="data/processed",
        threshold=0.01,
        min_interval=0,
        skip_frames=1,
        frame_scale=1,
        brightness_threshold=70
    )
    processor.process()
