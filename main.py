import sys


def main():
    try:
        image_path = sys.argv[1]
    except IndexError:
        print('Path to image should be passed')


if __name__ == '__main__':
    main()
