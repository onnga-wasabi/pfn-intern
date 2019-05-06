import json
import random
from collections import OrderedDict
from pathlib import Path

import chainer
import cv2
import numpy as np
from finger import setup_fingers
from settings import (
    ANNOTATED_LOG,
    COORDINATES_DIR,
    FINGER_TYPES,
    IMAGE_DIR,
    PRED_COORDINATES_DIR
)


def get_files(dir_path=IMAGE_DIR, log=ANNOTATED_LOG):
    img_dir = Path(dir_path)
    with open(log, 'r') as rf:
        execluded_files = [line.strip('\n') for line in rf]
    return [fname.resolve() for fname in img_dir.glob('*') if fname.name.split('.')[0] not in execluded_files]


def save_coordinates(fingers, fname, dir_path=COORDINATES_DIR, log=ANNOTATED_LOG):
    save_name = Path(fname.split('.')[0] + '.json')
    coordinates = OrderedDict({finger_type: finger.key_points for finger_type, finger in zip(FINGER_TYPES, fingers)})
    coordinates['fname'] = fname
    with open(dir_path / save_name, 'w') as wf:
        json.dump(coordinates, wf)
    if log is not None:
        with open(log, 'a') as af:
            af.write(fname.split('.')[0] + '\n')


def load_coordinates(fname, dir_path=PRED_COORDINATES_DIR):
    load_name = Path(fname.split('.')[0] + '.json')
    with open(dir_path / load_name, 'r') as rf:
        pred = json.load(rf)
    coordinates = [pred[finger_type] for finger_type in FINGER_TYPES]
    return coordinates


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


def save_predicted_images(model, dataset, files, gpu=-1, prefix=''):
    device = chainer.get_device(gpu)
    device.use()
    model.to_device(device)
    xp = chainer.cuda.cupy if gpu > -1 else np
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for data, fname in zip(dataset, files):
            img, _ = data
            heat_map = model.forward(xp.array([img]).astype('f'))[0]
            heat_map.to_cpu()

            fingers = setup_fingers()
            [finger.init_points() for finger in fingers]
            img = np.transpose(img, (1, 2, 0))
            finger_idx = 0
            for i, c in enumerate(heat_map):
                y, x = np.unravel_index(np.argmax(c.array), c.shape)
                y = int(y * (224 / 56))
                x = int(x * (224 / 56))
                fingers[finger_idx].add_point((x, y))
                if (i + 1) % 4 == 0:
                    for (x, y) in fingers[finger_idx].key_points:
                        img = cv2.drawMarker(img, (x, y), fingers[finger_idx].color, 0, 10)
                    for ((x1, y1), (x2, y2)) in fingers[finger_idx].edges:
                        img = cv2.line(img, (x1, y1), (x2, y2), fingers[finger_idx].color, 1, 0)
                    finger_idx += 1

            fname = fname.name.split('.')[0]
            cv2.imwrite(f'{prefix}pred_{fname}.png', img)


def save_predicted_coordinates(model, dataset, files, gpu=-1, prefix=''):
    device = chainer.get_device(gpu)
    device.use()
    model.to_device(device)
    xp = chainer.cuda.cupy if gpu > -1 else np
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for data, fname in zip(dataset, files):
            img, _ = data
            heat_map = model.forward(xp.array([img]).astype('f'))[0]
            heat_map.to_cpu()

            fingers = setup_fingers()
            [finger.init_points() for finger in fingers]
            img = np.transpose(img, (1, 2, 0))
            finger_idx = 0
            for i, c in enumerate(heat_map):
                y, x = np.unravel_index(np.argmax(c.array), c.shape)
                y = int(y * (224 / 56))
                x = int(x * (224 / 56))
                fingers[finger_idx].add_point((x, y))
                if (i + 1) % 4 == 0:
                    finger_idx += 1
            save_coordinates(fingers, fname, dir_path=PRED_COORDINATES_DIR, log=None)
