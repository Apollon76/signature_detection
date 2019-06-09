import cv2
import pandas as pd
import joblib

from signature_detection.classifier import Classifier


def main():
    data = pd.read_csv('../data/labels.csv')
    labels = data['has_signature'].apply(bool)
    names = data['name']
    images = [cv2.imread(f'../data/images/{name}.tif', 0) for name in names]
    classifier = Classifier()
    classifier.fit(images, labels)
    joblib.dump(classifier, 'classifier.joblib')


if __name__ == '__main__':
    main()
