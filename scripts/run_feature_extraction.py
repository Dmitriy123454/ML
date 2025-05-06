from src.utils.feature_extraction import TemporalFeatureEngineer
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    tfe = TemporalFeatureEngineer(metadata_path="data/processed/metadata.json")
    features = tfe.create_features("data/processed")
    np.save("data/processed/features.npy", features)
