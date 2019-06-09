import sys

import cv2
import joblib


def main():
    try:
        image_path = sys.argv[1]
    except IndexError:
        print('Path to image should be passed')
        return

    pipeline = joblib.load('signature_detection/classifier.joblib')
    print(pipeline.predict([cv2.imread(image_path, 0)]))


if __name__ == '__main__':
    main()
