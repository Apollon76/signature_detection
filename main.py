import sys

import numpy as np


def main():
    print(np.array([[1, 2, 3], [4, 5, 6]]).shape)
    try:
        image_path = sys.argv[1]
    except IndexError:
        print('Path to image should be passed')


if __name__ == '__main__':
    main()
