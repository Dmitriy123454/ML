import cv2
import os
import json
import numpy as np
import shutil
from datetime import datetime
from skimage.metrics import structural_similarity

class VideoProcessor:
    def __init__(self, video_path, output_dir="data/processed", threshold=0.01, min_interval=0, skip_frames=1, frame_scale=1, brightness_threshold=70):
        self.video_path = video_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.min_interval = min_interval
        self.skip_frames = skip_frames
        self.frame_scale = frame_scale
        self.brightness_threshold = brightness_threshold
        self.metadata = []

        # Очистка выходной папки перед стартом
        if os.path.exists(self.output_dir):
            self._clear_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)

    def _clear_output_dir(self):
        """Удаляет старые кадры в папке"""
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    os.remove(file_path)
                    print(f"🗑 Удалён файл: {file_path}")
                except Exception as e:
                    print(f"Ошибка удаления {file_path}: {e}")

    def _frame_diff(self, prev, curr):
        """Сравнение кадров через SSIM"""
        score, _ = structural_similarity(prev, curr, full=True)
        return score < (1 - self.threshold)

    def _frame_brightness(self, frame):
        """Оценивает среднюю яркость кадра"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        return brightness

    def _is_bright(self, frame):
        """Проверяет, является ли кадр достаточно ярким"""
        return self._frame_brightness(frame) > self.brightness_threshold

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ Не удалось открыть видеофайл: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.frame_scale)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.frame_scale)
        min_interval_frames = int(fps * self.min_interval)

        print(f"🎬 Начата обработка видео: {self.video_path}")
        print(f"Параметры обработки: порог изменений: {self.threshold}, интервал: {self.min_interval} сек, масштаб: {self.frame_scale}, яркость: >{self.brightness_threshold}")

        prev_frame = None
        frame_count = 0
        saved_count = 0
        last_saved_frame = -min_interval_frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.skip_frames != 0:
                frame_count += 1
                continue

            frame = cv2.resize(frame, (frame_width, frame_height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

            curr_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            if prev_frame is not None:
                time_diff = frame_count - last_saved_frame
                is_different = self._frame_diff(prev_frame, gray_blurred)
                is_bright = self._is_bright(frame)

                if time_diff > min_interval_frames and is_different and is_bright:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"frame_{timestamp}.jpg"
                    save_path = os.path.join(self.output_dir, filename)
                    cv2.imwrite(save_path, frame)

                    self.metadata.append({
                        "filename": filename,
                        "frame_number": frame_count,
                        "timestamp": curr_time_sec  # ❗ Добавляем сюда timestamp
                    })

                    saved_count += 1
                    last_saved_frame = frame_count
                    print(f"✅ Сохранён кадр: {filename}")
                else:
                    reasons = []
                    if not is_different:
                        reasons.append("похожий")
                    if not is_bright:
                        reasons.append("тёмный")
                    if reasons:
                        print(f"⏩ Кадр пропущен: {' и '.join(reasons)}")

            else:
                print("🚀 Первый кадр, сравнения нет")

            prev_frame = gray_blurred.copy()
            frame_count += 1

            if frame_count % int(fps * 5) == 0:
                current_time = frame_count / fps
                print(f"⏱ [{current_time:.1f} сек] Обработано: {frame_count} кадров, Сохранено: {saved_count}")

        cap.release()

        # Сохраняем метаданные
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=4)

        print(f"\n🎯 Завершено! Сохранено {saved_count} информационных кадров")
