import os

WINDOW_SIZE = 720
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 0, 0),
]

DEFAULT_POSITIONS = [  # based on 224x224
    [[112, 50], [150, 100], [150, 128], [150, 150]],
    [[112, 50], [125, 100], [125, 128], [125, 150]],
    [[112, 50], [112, 100], [112, 128], [112, 150]],
    [[112, 50], [95, 100], [95, 128], [95, 150]],
    [[112, 50], [75, 100], [75, 128], [75, 150]],
]

FINGER_TYPES = [
    'thumb',
    'index',
    'middle',
    'ring',
    'little',
]

SEED = 23121988

IMAGE_DIR = 'data/img/Hands'
COORDINATES_DIR = 'data/coordinates'
ANNOTATED_LOG = 'data/annotated.log'
if not os.path.exists(ANNOTATED_LOG):
    with open(ANNOTATED_LOG, 'w') as wf:
        wf.write('')
