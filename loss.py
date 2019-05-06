import chainer.functions as F


def weighted_mean_absolute_error(x, y, w_scale=1e-3):
    xp = x.xp
    weight = xp.where(y > 0, 1, w_scale)
    errors = F.absolute_error(x, y) * weight
    return F.mean(errors)
