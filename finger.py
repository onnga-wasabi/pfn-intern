from settings import COLORS, DEFAULT_POSITIONS


class Finger(object):

    def __init__(self, color=None):
        self.color = color or (0, 0, 0)
        self.init_points()

    def __len__(self):
        return len(self.key_points)

    def add_point(self, coordinate):
        self.key_points.append(coordinate)
        return self

    def init_points(self):
        self.key_points = []
        return self

    @property
    def edges(self):
        if len(self) > 1:
            return [(self.key_points[i], self.key_points[i + 1]) for i in range(len(self) - 1)]
        return []


def setup_fingers(colors=COLORS, coordinates=DEFAULT_POSITIONS):
    fingers = [Finger(c) for c in COLORS]
    for i, finger in enumerate(fingers):
        [finger.add_point(p) for p in coordinates[i]]
    return fingers
