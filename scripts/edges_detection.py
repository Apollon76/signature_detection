import cv2


def main():
    img = cv2.imread('../example.tif', 0)
    scale = 0.4
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    blur_size = 15
    blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    cv2.imshow('blur', blur)

    edges = cv2.Canny(blur, 100, 200)

    components_number, labels = cv2.connectedComponents(edges)

    components = [[] for i in range(components_number)]
    for i, row in enumerate(labels):
        for j, e in enumerate(row):
            if e != 0:
                components[e].append((i, j))

    print(len(components))
    print(components)

    cv2.imshow('edges', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
