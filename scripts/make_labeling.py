import os
from pathlib import Path

import cv2
import pandas as pd


def main():
    try:
        labels = pd.read_csv('labels.csv')
    except FileNotFoundError:
        labels = pd.DataFrame(columns=['name', 'has_signature'])

    names = set(labels['name'].values)
    print(names)
    path = Path('../data')
    q = 0
    max_height = 1200
    max_width = 1000
    for file in os.listdir(path):
        q += 1
        print('Number:', q)
        file = Path(file)
        print(file)

        name = file.stem
        if name in names:
            continue

        img = cv2.imread(str(path.joinpath(file)), cv2.IMREAD_GRAYSCALE)

        coef = min(max_width / img.shape[0], max_height / img.shape[1])
        img = cv2.resize(img, (0, 0), fx=coef, fy=coef)
        cv2.imshow('image', img)
        label = cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(label)
        if label == 27:
            break

        if label == 44:
            label = 0
        elif label == 46:
            label = 1
        else:
            break

        labels = pd.concat([
            labels,
            pd.DataFrame({'name': [name], 'has_signature': [label]})
        ], ignore_index=True)
        labels.to_csv('labels.csv', index=False)
    print(labels)


if __name__ == '__main__':
    main()
