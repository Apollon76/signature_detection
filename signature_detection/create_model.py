import cv2
import joblib
import pandas as pd
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from signature_detection.feature_extractor import FeatureExtractor


def main():
    data = pd.read_csv('../data/labels.csv')
    data = data[:300]
    labels = data['has_signature'].apply(bool)
    names = data['name']
    images = [cv2.imread(f'../data/images/{name}.tif', 0) for name in tqdm.tqdm(names)]

    feature_extractor = FeatureExtractor()
    classifier = RandomForestClassifier(n_estimators=20, random_state=42)
    pipeline = Pipeline([
        ('features', feature_extractor),
        ('classifier', classifier)
    ])
    pipeline.fit(images, labels)
    joblib.dump(pipeline, 'classifier.joblib')


if __name__ == '__main__':
    main()
