from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int

    def __lt__(self, other):
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y
