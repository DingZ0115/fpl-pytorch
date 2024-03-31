# 计算mean和std大小
import joblib
import torch
import numpy as np
from utils.dataset import SceneDatasetCV
from utils.generic import get_args

if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = "data"
    data = joblib.load(args.in_data)
    traj_len = data["trajectories"].shape[1]
    print(len(data["trajectories"]))
    train_splits = list(range(args.nb_splits))
    train_dataset = SceneDatasetCV(data, args.input_len, args.offset_len, args.pred_len,
                                   args.width, args.height, data_dir, train_splits, args.nb_train,
                                   True, "scale" in args.model, args.ego_type)
    print(train_dataset.X.shape)

    total_sum = torch.zeros(3, 1)
    total_sum_sq = torch.zeros(3, 1)  # 用于计算平方和
    cnt = 0

    for idx in range(len(train_dataset)):
        past, _ = train_dataset[idx][:2]  # 假设数据集返回的是(past, future)对
        total_sum += torch.sum(past, dim=0).unsqueeze(1)
        total_sum_sq += torch.sum(past ** 2, dim=0).unsqueeze(1)  # 累加当前批次的平方总和
        cnt += past.size(0)  # 累加样本数量

    # 计算整个数据集的平均值
    mean = total_sum / cnt

    # 计算整个数据集的方差，然后取平方根得到标准差
    # 方差 = 平方的平均值 - 平均值的平方
    variance = (total_sum_sq / cnt) - (mean ** 2)
    std = torch.sqrt(variance)

    print("Dataset Mean:\n", mean)
    print("Dataset Std Dev:\n", std)
