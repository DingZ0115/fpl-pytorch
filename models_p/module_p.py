import torch
import torch.nn as nn

import torch.nn.functional as F


class Conv_BN(nn.Module):
    def __init__(self, nb_in, nb_out, ksize=1, pad=0, no_bn=False):
        super(Conv_BN, self).__init__()
        self.no_bn = no_bn
        self.conv = nn.Conv1d(nb_in, nb_out, kernel_size=ksize, padding=pad)
        if not no_bn:
            self.bn = nn.BatchNorm1d(nb_out)

    def forward(self, x):
        if self.no_bn:
            return self.conv(x)
        else:
            return F.relu(self.bn(self.conv(x)))


class Conv_Module(nn.Module):
    def __init__(self, nb_in, nb_out, inter_list=None, no_act_last=False):
        super(Conv_Module, self).__init__()
        if inter_list is None:
            inter_list = []
        self.nb_layers = len(inter_list) + 1

        if len(inter_list) == 0:
            self.layer1 = Conv_BN(nb_in, nb_out, no_bn=no_act_last)
        else:
            self.layer1 = Conv_BN(nb_in, inter_list[0])
            for lidx, (nin, nout) in enumerate(zip(inter_list[:-1], inter_list[1:])):
                setattr(self, "layer{}".format(lidx + 2), Conv_BN(nin, nout))
            setattr(self, "layer{}".format(self.nb_layers), Conv_BN(inter_list[-1], nb_out, no_bn=no_act_last))

    def forward(self, h):
        for idx in range(1, self.nb_layers + 1):
            h = getattr(self, "layer{}".format(idx))(h)
        return h


class Encoder(nn.Module):
    def __init__(self, nb_inputs, channel_list, ksize_list, pad_list=None):
        super(Encoder, self).__init__()
        if pad_list is None:
            pad_list = []
        self.nb_layers = len(channel_list)
        channel_list = [nb_inputs] + channel_list
        if len(pad_list) == 0:
            pad_list = [0 for _ in range(len(ksize_list))]
        for idx, (nb_in, nb_out, ksize, pad) in enumerate(
                zip(channel_list[:-1], channel_list[1:], ksize_list, pad_list)):
            setattr(self, "conv{}".format(idx), Conv_BN(nb_in, nb_out, ksize, pad))

    def forward(self, x):
        h = torch.transpose(x, 1, 2)  # (B, D, L)
        for idx in range(self.nb_layers):
            h = getattr(self, "conv{}".format(idx))(h)
        return h


class Decoder(nn.Module):
    def __init__(self, nb_inputs, channel_list, ksize_list, no_act_last=False):
        super(Decoder, self).__init__()
        self.nb_layers = len(channel_list)
        self.no_act_last = no_act_last
        channel_list = channel_list + [nb_inputs]
        self.deconv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for idx, (nb_in, nb_out, ksize) in enumerate(zip(channel_list[:-1], channel_list[1:], ksize_list[::-1])):
            self.deconv_layers.append(nn.ConvTranspose1d(nb_in, nb_out, kernel_size=ksize))
            if not (no_act_last and idx == self.nb_layers - 1):
                self.bn_layers.append(nn.BatchNorm1d(nb_out))

    def forward(self, h):
        for idx in range(self.nb_layers):
            if self.no_act_last and idx == self.nb_layers - 1:
                h = self.deconv_layers[idx](h)
            else:
                h = F.relu(self.bn_layers[idx](self.deconv_layers[idx](h)))
        return h
