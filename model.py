import chainer
import chainer.functions as F
import chainer.links as L
from loss import weighted_mean_absolute_error


class PoseModelBase(chainer.Chain):

    def __init__(self, out):
        super(PoseModelBase, self).__init__()
        with self.init_scope():
            self.encoder = L.VGG16Layers()
            self.conv1_1 = L.Convolution2D(None, out, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(None, out, 3, 1, 1)

            self.conv2_1 = L.Convolution2D(None, out, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(None, out, 3, 1, 1)

            self.conv3_1 = L.Convolution2D(None, out, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(None, out, 3, 1, 1)

    def forward(self, x):
        h = self.base(x)
        h = self.refine1(h)
        h = self.refine2(h)
        return h

    def base(self, x):
        self.base_embeddings = self.encoder(x, layers=['conv3_3'])['conv3_3']
        h = F.relu(self.conv1_1(self.base_embeddings))
        h = F.relu(self.conv1_2(h))
        return h

    def refine1(self, h):
        h = h + F.relu(self.conv2_1(self.base_embeddings))
        h = F.relu(self.conv2_2(h))
        return h

    def refine2(self, h):
        h = h + F.relu(self.conv3_1(self.base_embeddings))
        h = F.relu(self.conv3_2(h))
        return h

    def refine_WMAE(self, x, y):
        h = self.base(x)
        loss_base = weighted_mean_absolute_error(h, y)

        h = self.refine1(h)
        loss_refine1 = weighted_mean_absolute_error(h, y)

        h = self.refine2(h)
        loss_refine2 = weighted_mean_absolute_error(h, y)

        loss = loss_base + loss_refine1 + loss_refine2
        chainer.report({
            'loss': loss,
            'loss_base': loss_base,
            'loss_refine1': loss_refine1,
            'loss_refine2': loss_refine2
        }, self)
        return loss
