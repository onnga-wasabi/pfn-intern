import argparse
import os
from datetime import datetime
from pathlib import Path

import chainer
from chainer import training
from chainer.training import extensions
from sklearn.model_selection import train_test_split

from dataset import PoseDataset
from model import PoseModelBase
from settings import COORDINATES_DIR, SEED
from util import set_random_seed


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-b', '--batch', type=int, default=10)
    parser.add_argument('-e', '--epoch', type=int, default=300)
    parser.add_argument('-lr', '--lr', type=float, default=1e-5)
    return parser.parse_args()


def snapshot_object_to_logdir(target, filename, savefun=chainer.serializers.npz.save_npz):
    @chainer.training.extension.make_extension(priority=-100)
    def snapshot_object_to_logdir(trainer):
        _snapshot_object_to_logdir(trainer, target, filename, savefun)

    return snapshot_object_to_logdir


def _snapshot_object_to_logdir(trainer, target, filename, savefun):
    savefun(filename, target)


def main():
    args = parse()
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    print(timestamp)
    set_random_seed(SEED)

    device = chainer.get_device(args.gpu)
    device.use()

    model = PoseModelBase(20)
    model.to_device(device)

    optimizer = chainer.optimizers.Adam(args.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    files = list(Path(COORDINATES_DIR).glob('*'))
    train_files, val_files = train_test_split(files, test_size=0.1, random_state=SEED)
    train = PoseDataset(train_files, augment=True)
    val = PoseDataset(val_files)

    train_iter = chainer.iterators.MultithreadIterator(train, args.batch, n_threads=8)
    test_iter = chainer.iterators.MultithreadIterator(val, args.batch, repeat=False, shuffle=False, n_threads=8)

    updater = chainer.training.updaters.StandardUpdater(train_iter, optimizer, loss_func=model.refine_WMAE, device=device)

    stop_trigger = (args.epoch, 'epoch')
    trainer = training.Trainer(updater, stop_trigger)
    os.mkdir(f'{trainer.out}/{timestamp}')

    trainer.extend(extensions.Evaluator(test_iter, model, eval_func=model.refine_WMAE, device=device))

    log_name = f'{timestamp}/log'
    trainer.extend(extensions.LogReport(filename=log_name))
    trainer.extend(extensions.PrintReport([
        'epoch',
        'main/loss_base',
        'main/loss_refine1',
        'main/loss_refine2',
        'validation/main/loss_base',
        'validation/main/loss_refine1',
        'validation/main/loss_refine2',
    ]), trigger=(1, 'epoch'))

    snapshot_trigger = training.triggers.MinValueTrigger('validation/main/loss_refine2', trigger=(10, 'epoch'))
    snapshot_name = f'{trainer.out}/{timestamp}/snapshot.npz'
    trainer.extend(snapshot_object_to_logdir(model, snapshot_name), trigger=snapshot_trigger)

    # trainer.extend(extensions.ProgressBar(update_interval=1))

    trainer.run()
    chainer.serializers.save_npz(f'{trainer.out}/{timestamp}/model_latest.npz', model)
    print(timestamp)


if __name__ == '__main__':
    main()
