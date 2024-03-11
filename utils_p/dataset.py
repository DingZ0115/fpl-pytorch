import torch
import numpy as np
from torch.utils.data import Dataset
import quaternion


def parse_data_CV(data, split_list, input_len, offset_len, pred_len, nb_train):
    if type(split_list) != list:
        split_list = [split_list]

    trajectories = torch.tensor(data["trajectories"], dtype=torch.float32)
    splits = torch.tensor(data["splits"], dtype=torch.long)
    traj_len = offset_len + pred_len
    idxs_past = torch.arange(offset_len - input_len, offset_len)
    idxs_pred = torch.arange(offset_len, traj_len)
    idxs_both = torch.arange(offset_len - input_len, traj_len)

    # Create a boolean mask for selecting splits
    # idxs_split = torch.tensor([splits == s for s in split_list]).any(0)
    idxs_split = torch.zeros_like(splits, dtype=torch.bool)
    for s in split_list:
        idxs_split |= (splits == s)

    # Select random indices if nb_train is specified
    if nb_train != -1:
        selected_indices = torch.randperm(idxs_split.sum()).numpy()[:nb_train]
    else:
        selected_indices = torch.arange(idxs_split.sum())

    # Use boolean indexing and select data based on the calculated indices
    # def select_data(x, indices):
    #     if x is None:
    #         return None
    #     else:
    #         return x[idxs_split][..., indices, :]
    #
    # data = [
    #     select_data(trajectories, idxs_past),
    #     select_data(trajectories, idxs_pred),
    #     data["video_ids"][idxs_split][selected_indices] if nb_train != -1 else data["video_ids"][idxs_split],
    #     data["frames"][idxs_split][selected_indices] if nb_train != -1 else data["frames"][idxs_split],
    #     data["person_ids"][idxs_split][selected_indices] if nb_train != -1 else data["person_ids"][idxs_split],
    #     select_data(torch.tensor(data["poses"], dtype=torch.float32), idxs_both),
    #     data["turn_mags"][idxs_split][selected_indices] if nb_train != -1 else data["turn_mags"][idxs_split],
    #     data["trans_mags"][idxs_split][selected_indices] if nb_train != -1 else data["trans_mags"][idxs_split],
    #     torch.tensor(data["masks"], dtype=torch.float32)[idxs_split][:, idxs_pred] if "masks" in data else None
    # ]
    def select_data(x, indices):
        if x is None:
            return None
        else:
            return x[:, indices, :]

    data = [
        select_data(trajectories, idxs_past),
        select_data(trajectories, idxs_pred),
        np.array(data["video_ids"][idxs_split]),
        np.array(data["frames"][idxs_split]),
        np.array(data["person_ids"][idxs_split]),
        select_data(torch.tensor(data["poses"], dtype=torch.float32), idxs_both),
        np.array(data["turn_mags"][idxs_split]),
        np.array(data["trans_mags"][idxs_split]),
        np.array(data["masks"][idxs_split][:, idxs_pred], dtype=np.float32) if "masks" in data else None
    ]

    return data + [offset_len - input_len]


def accumulate_egomotion(rots, vels):  # Accumulate translation and rotation
    egos = []
    qa = quaternion.from_float_array(np.array([1, 0, 0, 0], dtype=np.float32))
    va = torch.tensor([0., 0., 0.], dtype=torch.float32)
    for rot, vel in zip(rots, vels):
        rot_q = quaternion.from_rotation_vector(torch.tensor(rot, dtype=torch.float32))
        vel_rot = quaternion.rotate_vectors(rot_q, torch.tensor(vel, dtype=torch.float32))
        va += vel_rot
        qa = qa * rot_q
        ego = torch.cat((torch.tensor(quaternion.as_rotation_vector(qa), dtype=torch.float64), va), dim=0)
        egos.append(ego)
    return egos


class SceneDatasetCV(Dataset):
    def __init__(self, data, input_len, offset_len, pred_len, width, height,
                 data_dir, split_list, nb_train=-1, flip=False, use_scale=False, ego_type="sfm"):
        self.X, self.Y, self.video_ids, self.frames, self.person_ids, \
        raw_poses, self.turn_mags, self.trans_mags, self.masks, self.offset = \
            parse_data_CV(data, split_list, input_len, offset_len, pred_len, nb_train)

        # Convert to PyTorch tensors if not already
        raw_poses = torch.tensor(raw_poses, dtype=torch.float32)

        # Pose normalization and scale calculation
        past_len = input_len
        poses = raw_poses[..., :2]
        spine = (poses[..., 8:9, :] + poses[..., 11:12, :]) / 2
        neck = poses[..., 1:2, :]
        scales_all = torch.norm(neck - spine, dim=-1)  # (N, T, 1)
        scales_all = torch.clamp(scales_all, min=1e-8)  # Avoid ZeroDivisionError
        poses = (poses - spine) / scales_all.unsqueeze(-1)  # Normalization

        self.poses = poses
        self.scales_all = scales_all[..., 0]
        self.scales = self.scales_all[:, -pred_len - 1]

        # (x, y) -> (x, y, s)
        if use_scale:
            self.X = torch.cat((self.X, self.scales_all[:, :past_len, None]), dim=-1)
            self.Y = torch.cat((self.Y, self.scales_all[:, past_len:past_len + pred_len, None]), dim=-1)

        self.width = width
        self.height = height
        self.data_dir = data_dir
        self.flip = flip
        self.nb_inputs = self.X.shape[-1]
        self.ego_type = ego_type

        self.egomotions = []
        for vid, frame in zip(self.video_ids, self.frames):
            ego_dict = data["egomotion_dict"][vid]
            if ego_type == "sfm":  # SfMLearner
                rots, vels = [], []
                for f in range(frame + self.offset, frame + self.offset + past_len + pred_len):
                    key = f"rgb_{f:05d}.jpg"
                    key_m1 = f"rgb_{f - 1:05d}.jpg"
                    rot_vel = torch.tensor(ego_dict[key] if key in ego_dict
                                           else ego_dict[key_m1] if key_m1 in ego_dict
                    else np.zeros(6), dtype=torch.float32)
                    rots.append(rot_vel[:3])
                    vels.append(rot_vel[3:])
                egos = accumulate_egomotion(torch.stack(rots[:past_len]), torch.stack(vels[:past_len])) + \
                       accumulate_egomotion(torch.stack(rots[past_len:past_len + pred_len]),
                                            torch.stack(vels[past_len:past_len + pred_len]))
            else:  # Grid optical flow
                raw_egos = [torch.tensor(ego_dict[f"rgb_{f:05d}.jpg"], dtype=torch.float32) for f in
                            range(frame + self.offset, frame + self.offset + past_len + pred_len)]
                egos = [torch.sum(torch.stack(raw_egos[:idx + 1]), dim=0) for idx in range(past_len)] + \
                       [torch.sum(torch.stack(raw_egos[past_len:past_len + idx + 1]), dim=0) for idx in range(pred_len)]
            self.egomotions.append(torch.stack(egos))
        self.egomotions = torch.stack(self.egomotions)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        # Clone the tensors to avoid modifying the original data
        X = self.X[i].clone()
        Y = self.Y[i].clone()
        poses = self.poses[i].clone()
        egomotions = self.egomotions[i].clone()

        # Random horizontal flip
        horizontal_flip = torch.rand(1).item() < 0.5 if self.flip else False
        if horizontal_flip:
            X[:, 0] = self.width - X[:, 0]
            Y[:, 0] = self.width - Y[:, 0]
            poses[:, :, 0] = -poses[:, :, 0]
            if self.ego_type == "sfm":
                egomotions[:, [1, 2, 3]] = -egomotions[:, [1, 2, 3]]
            else:
                # The aim is to flip every other dimension starting from 0, assuming egomotions is a 2D tensor.
                nb_dims = egomotions.shape[1]
                for j in range(0, nb_dims, 2):
                    egomotions[:, j] = -egomotions[:, j]

        return X, Y, poses, self.video_ids[i], self.frames[i], self.person_ids[i], \
               horizontal_flip, egomotions, self.scales[i], self.turn_mags[i], self.scales_all[i]


class SceneDatasetForAnalysis(Dataset):
    """
    Dataset class only for plot
    """

    def __init__(self, data, input_len, offset_len, pred_len, width, height):
        self.X, self.Y, self.video_ids, self.frames, self.person_ids, \
        raw_poses, self.turn_mags, self.trans_mags, self.masks, self.offset = \
            parse_data_CV(data, list(range(5, 10, 1)), input_len, offset_len, pred_len, -1)

        # Convert to PyTorch Tensors
        raw_poses = torch.tensor(raw_poses, dtype=torch.float32)

        # (N, T, D, 3)
        past_len = input_len
        poses = raw_poses[..., :2]
        spine = (poses[..., 8:9, :] + poses[..., 11:12, :]) / 2
        neck = poses[..., 1:2, :]
        sizes = torch.norm(neck - spine, dim=-1)  # (N, T, 1)
        poses = (poses - spine) / sizes.unsqueeze(-1)  # Normalization

        self.poses = poses
        self.sizes = sizes[..., 0]
        self.scales = self.sizes[:, -pred_len - 1]

        self.X = torch.cat((self.X, self.sizes[:, :past_len, None]), dim=-1)
        self.Y = torch.cat((self.Y, self.sizes[:, past_len:past_len + pred_len, None]), dim=-1)

        self.width = width
        self.height = height
        self.nb_inputs = self.X.shape[-1]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        X = self.X[i].clone()
        Y = self.Y[i].clone()
        poses = self.poses[i].clone()

        return X, Y, poses, self.video_ids[i], self.frames[i], self.person_ids[i], \
               self.scales[i], self.turn_mags[i], self.sizes[i]
