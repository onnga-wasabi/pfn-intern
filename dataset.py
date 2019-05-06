import json
import random

import chainer
import cv2
import numpy as np
from PIL import Image
from settings import IMAGE_DIR


class PoseDataset(chainer.dataset.DatasetMixin):

    intensity = 255
    in_size = 224
    out_size = 56

    def __init__(self, files, augment=False):
        self.dataset = files
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        with open(self.dataset[i], 'r') as rf:
            data = json.load(rf)
        fname = data['fname']
        img = Image.open(f'{IMAGE_DIR}/{fname}')
        img = chainer.links.model.vision.vgg.prepare(img)

        points = np.vstack((data['thumb'], data['index'], data['middle'], data['ring'], data['little']))
        heat_map = np.array([self.point2map(point) for point in points]).astype('f')

        if self.augment:
            img, heat_map = self.random_horizontal_flip(img, heat_map)
            img, heat_map = self.random_crop(img, heat_map)

        return img, heat_map

    def point2map(self, point):
        x, y = point
        heat_map = np.zeros((self.out_size, self.out_size))
        x = int(x * (self.out_size / self.in_size))
        y = int(y * (self.out_size / self.in_size))
        heat_map[y, x] = self.intensity
        return heat_map

    def random_horizontal_flip(self, img, heat_map):
        if random.choice([True, False]):
            img = img[:, :, ::-1]
            heat_map = heat_map[:, :, ::-1]
        return img, heat_map

    def random_crop(self, img, heat_map, crop_size=50):
        C, H, W = heat_map.shape
        x_offset = random.randint(0, W - crop_size)
        y_offset = random.randint(0, H - crop_size)
        x_slice = range(x_offset, x_offset + crop_size)
        y_slice = range(y_offset, y_offset + crop_size)
        if np.sum(heat_map[:, y_slice, x_slice]) == (self.intensity * 20):
            heat_map = np.array([cv2.resize(c[y_slice, x_slice], (self.out_size, self.out_size)) for c in heat_map])

            _, img_H, img_W = img.shape
            crop_size = int(crop_size * (self.in_size / self.out_size))
            x_offset = int(x_offset * (self.in_size / self.out_size))
            y_offset = int(y_offset * (self.in_size / self.out_size))
            x_slice = range(x_offset, x_offset + crop_size)
            y_slice = range(y_offset, y_offset + crop_size)
            img = np.array([cv2.resize(c[y_slice, x_slice], (self.in_size, self.in_size)) for c in img])

        return img, heat_map
