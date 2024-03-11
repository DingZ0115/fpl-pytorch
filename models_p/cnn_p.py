import torch
import torch.nn as nn
from logging import getLogger

logger = getLogger("main")

from models.module import Conv_Module, Encoder, Decoder


class CNNBase(nn.Module):
    def __init__(self, mean, std, gpu):
        super(CNNBase, self).__init__()
        self._mean = mean
        self._std = std
        self.nb_inputs = len(mean)
        self.target_idx = -1

        # Send mean and std of the dataset to GPU to produce prediction result in the image coordinate
        if torch.cuda.is_available():
            self.mean = mean.clone().detach().cuda(gpu)
            self.std = std.clone().detach().cuda(gpu)
            # self.mean = torch.tensor(mean, dtype=torch.float32).cuda(gpu)
            # self.std = torch.tensor(std, dtype=torch.float32).cuda(gpu)
        else:
            self.mean = mean.clone().detach()
            self.std = std.clone().detach()
            # self.mean = torch.tensor(mean, dtype=torch.float32)
            # self.std = torch.tensor(std, dtype=torch.float32)

    def _prepare_input(self, inputs):
        pos_x, pos_y, poses, egomotions = inputs[:4]
        if pos_y.data.ndim == 2:
            pos_x = torch.unsqueeze(pos_x, 0)
            pos_y = torch.unsqueeze(pos_y, 0)
            if egomotions is not None:
                egomotions = torch.unsqueeze(egomotions, 0)
            poses = torch.unsqueeze(poses, 0)

        # Locations
        # Note: prediction target is displacement from last input
        x = (pos_x - self.mean.expand_as(pos_x)) / self.std.expand_as(pos_x)
        y = (pos_y - self.mean.expand_as(pos_y)) / self.std.expand_as(pos_y)
        y = y - x[:, -1:, :].expand_as(pos_y)

        # Egomotions
        past_len = pos_x.shape[1]
        if egomotions is not None:
            ego_x = egomotions[:, :past_len, :]
            ego_y = egomotions[:, past_len:, :]

        # Poses
        poses = poses.view(poses.shape[0], poses.shape[1], -1)
        pose_x = poses[:, :past_len, :]
        pose_y = poses[:, past_len:, :]

        if egomotions is not None:
            return x, y, x[:, -1, :], ego_x, ego_y, pose_x, pose_y
        else:
            return x, y, x[:, -1, :], None, None, pose_x, pose_y

    def predict(self, inputs):
        return self(inputs)


class CNN(CNNBase):
    def __init__(self, mean, std, device, channel_list, dc_channel_list, ksize_list, dc_ksize_list, inter_list,
                 last_list, pad_list):
        super(CNN, self).__init__(mean, std, device)
        # Ensure dc_ksize_list is not empty and defaults to ksize_list if it is
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list

        # Initialize network components
        self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list).to(device)
        self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, list(reversed(dc_ksize_list))).to(device)
        self.inter = Conv_Module(channel_list[-1], dc_channel_list[0], inter_list).to(device)
        self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True).to(device)

    def forward(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(inputs)
        h = self.pos_encoder(pos_x)
        h = self.inter(h)
        h = self.pos_decoder(h)
        pred_y = self.last(h)
        pred_y = torch.transpose(pred_y, 1, 2)
        pred_y = pred_y[:, :pos_y.shape[1], :]
        # loss = F.mse_loss(pred_y, pos_y)

        pred_y = (pred_y + offset_x.unsqueeze(
            1)) * self.std + self.mean  # Adjust for broadcasting and operations in PyTorch
        # return loss, pred_y
        return pred_y, pos_y


class CNN_Ego(CNNBase):
    """
    Baseline: feeds locations and egomotions
    """

    def __init__(self, mean, std, device, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list, ego_type):
        super(CNN_Ego, self).__init__(mean, std, device)
        ego_dim = 6 if ego_type == "sfm" else 96 if ego_type == "grid" else 24
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list

        self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list).to(device)
        self.ego_encoder = Encoder(ego_dim, channel_list, ksize_list, pad_list).to(device)
        self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, list(reversed(dc_ksize_list))).to(device)
        self.inter = Conv_Module(channel_list[-1] * 2, dc_channel_list[0], inter_list).to(device)
        self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True).to(device)

    def forward(self, inputs):
        pos_x, pos_y, offset_x, ego_x, _, _, _ = self._prepare_input(inputs)

        h_pos = self.pos_encoder(pos_x)
        h_ego = self.ego_encoder(ego_x)
        h = torch.cat((h_pos, h_ego), dim=1)  # PyTorch 使用 dim 而不是 axis
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_y = self.last(h_pos)
        pred_y = pred_y.transpose(1, 2)  # PyTorch 中转置张量的方法
        pred_y = pred_y[:, :, :pos_y.size(2)]  # 调整预测的形状以匹配目标
        pred_y = (pred_y * self.std) + self.mean
        return pred_y, pos_y


class CNN_Pose(CNNBase):
    """
    Baseline: feeds locations and poses
    """

    def __init__(self, mean, std, gpu, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(CNN_Pose, self).__init__(mean, std, gpu)
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
        self.pose_encoder = Encoder(36, channel_list, ksize_list, pad_list)
        self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
        self.inter = Conv_Module(channel_list[-1] * 2, dc_channel_list[0], inter_list)
        self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def forward(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(inputs)

        h_pos = self.pos_encoder(pos_x)
        h_pose = self.pose_encoder(pose_x)
        h = torch.cat((h_pos, h_pose), dim=1)  # (B, C, 2)
        h = self.inter(h)
        h_pos = self.pos_decoder(h)

        pred_y = self.last(h_pos)
        pred_y = pred_y.transpose(1, 2)  # PyTorch 中转置张量的方法
        pred_y = pred_y[:, :, :pos_y.size(2)]  # 调整预测的形状以匹配目标
        # 调整预测值，根据均值和标准差逆归一化
        pred_y = (pred_y * self.std) + self.mean
        return pred_y, pos_y


class CNN_Ego_Pose(CNNBase):
    """
    Our full model: feeds locations, egomotions, and poses as input
    """

    def __init__(self, mean, std, device, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list, ego_type):
        super(CNN_Ego_Pose, self).__init__(mean, std, device)
        ego_dim = 6 if ego_type == "sfm" else 96 if ego_type == "grid" else 24

        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list

        self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list).to(device)
        self.ego_encoder = Encoder(ego_dim, channel_list, ksize_list, pad_list).to(device)
        self.pose_encoder = Encoder(36, channel_list, ksize_list, pad_list).to(device)
        self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, list(reversed(dc_ksize_list))).to(device)
        self.inter = Conv_Module(channel_list[-1] * 3, dc_channel_list[0], inter_list).to(device)
        self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True).to(device)

    def forward(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(inputs)

        h_pos = self.pos_encoder(pos_x)
        h_pose = self.pose_encoder(pose_x)
        h_ego = self.ego_encoder(ego_x)

        h = torch.cat((h_pos, h_pose, h_ego), dim=1)  # PyTorch 使用 dim 而不是 axis
        h = self.inter(h)
        h_pos = self.pos_decoder(h)

        pred_y = self.last(h_pos)
        pred_y = pred_y.transpose(1, 2)  # PyTorch 中转置张量的方法
        pred_y = pred_y[:, :, :pos_y.size(2)]  # 调整预测的形状以匹配目标
        # 调整预测值，根据均值和标准差逆归一化
        pred_y = (pred_y * self.std) + self.mean
        return pred_y, pos_y

