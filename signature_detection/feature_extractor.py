from typing import List

import numpy as np
import cv2
from tqdm import tqdm

from signature_detection.point import Point
from signature_detection.utils import get_bounding_box


class FeatureExtractor:
    def fit(self, data, y):
        pass

    def transform(self, images) -> np.ndarray:
        data = np.array([self.extract_from_image(image) for image in tqdm(images)])
        data[data == np.inf] = 0
        data[data == -np.inf] = 0
        data[data == np.nan] = 0
        return data

    def fit_transform(self, data, y):
        self.fit(data, y)
        return self.transform(data)

    def extract_from_image(self, img) -> List[float]:
        scale = 0.3
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        blur_size = 9
        blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

        edges = cv2.Canny(blurred, 100, 200, L2gradient=True)
        components = self.__get_components(edges)

        main_candidate = max(components, key=lambda component: self.__get_avg_dist(component))
        left, right, top, bottom = get_bounding_box(main_candidate)
        return [
            self.__restrict(self.__get_avg_dist(main_candidate)),
            self.__restrict((top - bottom) / (right - left)),
            self.__restrict((top + bottom) / blurred.shape[0]),
        ]

    @staticmethod
    def __get_components(edges):
        components_number, labels = cv2.connectedComponents(edges)

        components = [[] for _ in range(components_number)]
        for i, row in enumerate(labels):
            for j, e in enumerate(row):
                if e != 0:
                    components[e].append(Point(i, j))
        return components

    @staticmethod
    def __get_avg_dist(component):
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

    @staticmethod
    def __restrict(x: float) -> float:
        return min(10 ** 9, max(-10 ** 9, x))
