from typing import List

from signature_detection.point import Point


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
