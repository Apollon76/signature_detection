from typing import List

import numpy as np
import cv2

from signature_detection.point import Point

eps = 10000


def count_salience(component: List[Point], img):
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    result = 0.0
    for i in range(len(component)):
        x1, y1 = component[i].x, component[i].y
        p1, q1 = grad_x[x1, y1], grad_y[x1, y1]
        for j in range(i):
            x2, y2 = component[j].x, component[j].y
            p2, q2 = grad_x[x2, y2], grad_y[x2, y2]
            den = (p1 * q2 - p2 * q1) ** 2
            if abs(den) < eps:
                continue
            result += 4 * (p1 * (x2 - x1) + q1 * (y2 - y1)) * (p2 * (x1 - x2) + q2 * (y1 - y2)) / den
    return result


def get_bounding_box(component: List[Point]):
    left = float('inf')
    right = -float('inf')
    top = -float('inf')
    bottom = float('inf')
    for point in component:
        left = min(left, point.x)
        right = max(right, point.x)
        top = max(top, point.y)
        bottom = min(bottom, point.y)
    return left, right, top, bottom


def get_avg_dist(component):
    if not component:
        return 0
    left, right, top, bottom = get_bounding_box(component)
    total_dist = 0
    for point in component:
        dist = abs(point.x - left)
        dist = min(dist, abs(point.x - right))
        dist = min(dist, abs(point.y - top))
        dist = min(dist, abs(point.y - bottom))
        total_dist += dist
    return total_dist / len(component)


def main():
    # img = cv2.imread('../another_image.jpg', 0)
    # img = cv2.imread('../cropped.png', 0)
    # img = cv2.imread('../bad_quality.jpg', 0)
    # img = cv2.imread('../many_signatures.jpg', 0)
    # img = cv2.imread('../data/00046211.tif', 0)
    img = cv2.imread('../example.tif', 0)
    print(img.shape)
    scale = 0.3
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    blur_size = 9
    blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    cv2.imshow('blur', blur)

    edges = cv2.Canny(blur, 100, 200, L2gradient=True)

    components_number, labels = cv2.connectedComponents(edges)

    components = [[] for _ in range(components_number)]
    for i, row in enumerate(labels):
        for j, e in enumerate(row):
            if e != 0:
                components[e].append(Point(i, j))

    print(len(components))
    print(components)
    avg_dists = []
    for component in components:
        avg_dists.append(get_avg_dist(component))
    max_value = max(avg_dists)
    avg_dist_img = np.zeros(blur.shape)
    for salience, component in sorted(zip(avg_dists, components), reverse=True)[:5]:
        for point in component:
            avg_dist_img[point.x, point.y] = 255 * salience / max_value
    cv2.imshow('avg_dists', avg_dist_img)
    # saliences = []
    # for component in components:
    #     saliences.append(count_salience(component, blur))
    # print(saliences)
    # saliences = [salience for salience in saliences]
    # max_value = max(saliences)
    # saliences_img = np.zeros(blur.shape)
    # print(max_value)

    # for salience, component in sorted(zip(saliences, components))[:5]:
    #     for point in component:
    #         saliences_img[point.x, point.y] = 255 * salience / max_value
    # cv2.imshow('saliences', saliences_img)

    cv2.imshow('edges', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
