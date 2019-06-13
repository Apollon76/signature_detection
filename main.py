import sys

import cv2
import joblib
import pandas as pd
import tqdm
from sklearn.metrics import f1_score, classification_report, roc_auc_score


def main():
    # try:
    #     image_path = sys.argv[1]
    # except IndexError:
    #     print('Path to image should be passed')
    #     return

    data = pd.read_csv('data/labels.csv')
    data = data[300:500]
    labels = data['has_signature'].apply(bool)
    names = data['name']
    images = [cv2.imread(f'data/images/{name}.tif', 0) for name in tqdm.tqdm(names)]

    pipeline = joblib.load('signature_detection/classifier.joblib')
    # result = pipeline.predict([cv2.imread(image_path, 0)])
    result = pipeline.predict(images)
    print(f1_score(labels, result, average='weighted'))
    # print('ROC-AUC:', roc_auc_score(labels, pipeline.predict_proba(images), average='weighted'))
    print(classification_report(labels, result))
    print(result)


if __name__ == '__main__':
    main()
