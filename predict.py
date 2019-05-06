import argparse
import os
from pathlib import Path

import chainer
from dataset import PoseDataset
from model import PoseModelBase
from settings import COORDINATES_DIR, SEED
from sklearn.model_selection import train_test_split
from util import save_predicted_images

TRAIN_RESULT = 'result'


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-t', '--timestamp')
    return parser.parse_args()


def main():
    args = parse()

    timestamp = args.timestamp
    model = PoseModelBase(20)
    snapshot_name = f'{TRAIN_RESULT}/{timestamp}/snapshot.npz'
    chainer.serializers.load_npz(snapshot_name, model)

    files = list(Path(COORDINATES_DIR).glob('*'))
    train_files, val_files = train_test_split(files, test_size=0.1, random_state=SEED)
    train = PoseDataset(train_files)
    val = PoseDataset(val_files)

    outdir = f'{TRAIN_RESULT}/{timestamp}/img'
    os.mkdir(outdir)
    save_predicted_images(model, train, train_files, args.gpu, prefix=f'{outdir}/')
    save_predicted_images(model, val, val_files, args.gpu, f'{outdir}/val_')
    print('output:', outdir)


if __name__ == '__main__':
    main()
