import torch
import torch.nn as nn

import torch.nn.functional as F


class Conv_BN(nn.Module):
    def __init__(self, nb_in, nb_out, ksize=1, pad=0, no_bn=False):
        super(Conv_BN, self).__init__()
        self.no_bn = no_bn
        self.conv = nn.Conv1d(nb_in, nb_out, kernel_size=ksize, padding=pad)
        # self.conv.weight.data.fill_(1)
        # self.conv.bias.data.fill_(0)
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
        self.layers = nn.ModuleList()
        # Initialize the first layer
        if len(inter_list) == 0:
            # Directly create a single layer with potentially no activation in the last layer
            self.layers.append(Conv_BN(nb_in, nb_out, no_bn=no_act_last))
        else:
            # First layer
            self.layers.append(Conv_BN(nb_in, inter_list[0]))
            # Intermediate layers
            for nin, nout in zip(inter_list[:-1], inter_list[1:]):
                self.layers.append(Conv_BN(nin, nout))
            # Last layer with potentially no activation
            self.layers.append(Conv_BN(inter_list[-1], nb_out, no_bn=no_act_last))

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


class Encoder(nn.Module):
    def __init__(self, nb_inputs, channel_list, ksize_list, pad_list=None):
        super(Encoder, self).__init__()
        if pad_list is None:
            pad_list = []
        channel_list = [nb_inputs] + channel_list
        if len(pad_list) == 0:
            pad_list = [0 for _ in range(len(ksize_list))]  # ensure padding list is correctly initialized
        self.layers = nn.ModuleList()  # Initialize ModuleList to hold layers
        for nb_in, nb_out, ksize, pad in zip(channel_list[:-1], channel_list[1:], ksize_list, pad_list):
            self.layers.append(Conv_BN(nb_in, nb_out, ksize, pad))  # Add layers to ModuleList

    def forward(self, x):
        h = torch.transpose(x, 1, 2)
        for layer in self.layers:
            h = layer(h)  # Pass the input through each layer in sequence
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
            conv_transpose_layer = nn.ConvTranspose1d(nb_in, nb_out, kernel_size=ksize)
            # conv_transpose_layer.weight.data.fill_(1)
            # conv_transpose_layer.bias.data.fill_(0)
            self.deconv_layers.append(conv_transpose_layer)
            if not (no_act_last and idx == self.nb_layers - 1):
                self.bn_layers.append(nn.BatchNorm1d(nb_out))

    def forward(self, h):
        for idx in range(self.nb_layers):
            deconv = self.deconv_layers[idx](h)
            if self.no_act_last and idx == self.nb_layers - 1:
                h = deconv
            else:
                h = F.relu(self.bn_layers[idx](deconv))
        return h
