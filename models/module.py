from __future__ import print_function
from __future__ import division
from six.moves import range

import chainer
import chainer.functions as F
import chainer.links as L


class Conv_BN(chainer.Chain):
    def __init__(self, nb_in, nb_out, ksize=1, pad=0, no_bn=False):
        super(Conv_BN, self).__init__()
        self.no_bn = no_bn
        with self.init_scope():
            self.conv = L.ConvolutionND(1, nb_in, nb_out, ksize=ksize, pad=pad)
            if not no_bn:
                self.bn = L.BatchNormalization(nb_out)

    def __call__(self, x):
        if self.no_bn:
            return self.conv(x)
        else:
            return F.relu(self.bn(self.conv(x)))


class Conv_Module(chainer.Chain):
    def __init__(self, nb_in, nb_out, inter_list=[], no_act_last=False):
        super(Conv_Module, self).__init__()
        self.nb_layers = len(inter_list) + 1
        with self.init_scope():
            if len(inter_list) == 0:
                setattr(self, "layer1", Conv_BN(nb_in, nb_out, no_bn=no_act_last))
            else:
                setattr(self, "layer1", Conv_BN(nb_in, inter_list[0]))
                for lidx, (nin, nout) in enumerate(zip(inter_list[:-1], inter_list[1:])):
                    setattr(self, "layer{}".format(lidx + 2), Conv_BN(nin, nout))
                setattr(self, "layer{}".format(self.nb_layers), Conv_BN(inter_list[-1], nb_out, no_bn=no_act_last))

    def __call__(self, h):
        for idx in range(1, self.nb_layers + 1, 1):
            h = getattr(self, "layer{}".format(idx))(h)
        return h


class Encoder(chainer.Chain):
    def __init__(self, nb_inputs, channel_list, ksize_list, pad_list=[]):
        super(Encoder, self).__init__()
        self.nb_layers = len(channel_list)
        channel_list = [nb_inputs] + channel_list
        if len(pad_list) == 0:
            pad_list = [0 for _ in range(len(ksize_list))]
        for idx, (nb_in, nb_out, ksize, pad) in enumerate(
                zip(channel_list[:-1], channel_list[1:], ksize_list, pad_list)):
            self.add_link("conv{}".format(idx), Conv_BN(nb_in, nb_out, ksize, pad))

    def __call__(self, x):
        h = F.swapaxes(x, 1, 2)  # (B, D, L)
        for idx in range(self.nb_layers):
            h = getattr(self, "conv{}".format(idx))(h)
        return h


class Decoder(chainer.Chain):
    def __init__(self, nb_inputs, channel_list, ksize_list, no_act_last=False):
        super(Decoder, self).__init__()
        self.nb_layers = len(channel_list)
        self.no_act_last = no_act_last
        channel_list = channel_list + [nb_inputs]
        for idx, (nb_in, nb_out, ksize) in enumerate(zip(channel_list[:-1], channel_list[1:], ksize_list[::-1])):
            self.add_link("deconv{}".format(idx), L.DeconvolutionND(1, nb_in, nb_out, ksize))
            if no_act_last and idx == self.nb_layers - 1:
                continue
            self.add_link("bn{}".format(idx), L.BatchNormalization(nb_out))

    def __call__(self, h):
        for idx in range(self.nb_layers):
            if self.no_act_last and idx == self.nb_layers - 1:
                h = getattr(self, "deconv{}".format(idx))(h)
            else:
                h = F.relu(getattr(self, "bn{}".format(idx))(getattr(self, "deconv{}".format(idx))(h)))
        return h
