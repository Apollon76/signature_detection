import os
import shutil
from pathlib import Path


def main():
    path = '../imagesa'
    target = Path('./data')
    try:
        os.mkdir(target)
    except FileExistsError:
        pass

    for r, d, files in os.walk(path):
        for filename in files:
            file = Path(r).joinpath(filename)
            if file.suffix == '.tif':
                shutil.copyfile(file, target.joinpath(filename))
                print(file)


if __name__ == '__main__':
    main()
