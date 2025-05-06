from src.utils.video_processing import VideoProcessor

if __name__ == "__main__":
    processor = VideoProcessor(
        video_path="data/raw/video.mp4",
        output_dir="data/processed",
        threshold=0.01,          # высокая чувствительность к изменениям
        min_interval=0,          # можно сохранять подряд
        skip_frames=1,           # без пропуска
        frame_scale=1,           # оригинальный размер
        brightness_threshold=70  # проверка на минимальную яркость
    )
    processor.process()
