import os
import cv2
import json
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern

class FeatureExtractor:
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False, pooling='avg')

    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        vgg_feat = self.model.predict(x).flatten()

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist = np.histogram(lbp, bins=20, range=(0, 10))[0]

        return np.concatenate([vgg_feat, lbp_hist])

class TemporalFeatureEngineer:
    def __init__(self, metadata_path):
        with open(metadata_path) as f:
            self.metadata = json.load(f)

    def create_features(self, img_dir):
        extractor = FeatureExtractor()
        features = []

        for m in self.metadata:
            img_path = os.path.join(img_dir, m['filename'])
            frame = cv2.imread(img_path)

            vis_feat = extractor.extract_features(img_path)
            time_feat = [m['frame_number'] / len(self.metadata)]

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])

            features.append(np.concatenate([vis_feat, [brightness], time_feat]))

        return StandardScaler().fit_transform(features)
