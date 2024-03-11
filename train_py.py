# The main training file, afer calling run.py to generate commands.
# This file will be called to start the main training parts.
# Look at the command file in gen_scripts for input parameters.

# python -u train_cv.py --root_dir experiments/5fold_proposed_only_240303_130532
# --in_data dataset/id_list_20_180521_173817_20_sp5_2_5.joblib --nb_iters 17000
# --iter_snapshot 1000 --optimizer adam --height 960 --batch_size 64 --save_model
# --nb_train -1 --pred_len 10 --channel_list 32 64 128 128 --deconv_list 256 128 64 32
# --ksize_list 3 3 3 3 --inter_list 256 --input_len 10 --lr_step_list 5000 10000 15000
# --model cnn_ego_pose_scale --nb_splits 5 --eval_split 4 --gpu 0

import os
import time
import json
import joblib
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR  # 假设你有类似的学习率调度需求
import torch.nn.functional as F
import numpy as np
from utils.dataset import SceneDatasetCV  # 假设你已经有一个对应的PyTorch Dataset实现
from utils.evaluation import Evaluator  # 需要根据PyTorch进行适当修改
from utils.summary_logger import SummaryLogger
from utils.generic import get_args, get_model, write_prediction
from mllogger import MLLogger

logger = MLLogger(init=False)


# def concat_examples_pytorch(batch, device, data_idxs):
#     batch_array = []
#     # batch 11*batch_size
#     for idx in data_idxs:
#         # 提取索引对应的样本并将它们移动到指定设备
#         # samples = [x[idx].to(device) for x in batch]
#         print(idx)
#         print(len(batch))
#         print(len(batch[0]))
#         samples = []
#         for x in batch:
#             print(len(x[idx]))
#             print(type(x[idx]))
#             sample = x[idx].to(device)
#             samples.append(sample)
#         batch_array.append(torch.cat(samples, dim=0))
#
#     return batch_array
def concat_examples_pytorch(batch, device, data_idxs):
    batch_array = []
    # batch 11*batch_size
    for idx in data_idxs:
        batch_array.append(batch[idx].to(device))
    return batch_array


def validate_and_report(model, valid_loader, device, args, save_dir, iter_cnt):
    prediction_dict = {
        "arguments": vars(args),
        "predictions": {}
    }
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在评估时不计算梯度
        model.zero_grad()  # 清除旧的梯度
        # 实现验证逻辑
        for batch in valid_loader:
            inputs = [batch[idx].to(device) for idx in data_idxs]
            # inputs = [tensor.to(args.device) for tensor in batch]
            pred_y, pos_y = model(inputs)
            loss = F.mse_loss(pred_y, pos_y)  # 计算损失
            # 更新验证评估
            valid_eval.update(loss.item(), pred_y, batch)

            write_prediction(prediction_dict["predictions"], batch, pred_y)

        message_str = "Iter {}: train loss {} / ADE {} / FDE {}, valid loss {} / " \
                      "ADE {} / FDE {}, elapsed time: {} (s)"
        logger.info(
            message_str.format(iter_cnt + 1, train_eval("loss"), train_eval("ade"), train_eval("fde"),
                               valid_eval("loss"), valid_eval("ade"), valid_eval("fde"), time.time() - st)
        )
        train_eval.update_summary(summary, iter_cnt, ["loss", "ade", "fde"])
        valid_eval.update_summary(summary, iter_cnt, ["loss", "ade", "fde"])

        predictions = prediction_dict["predictions"]
        pred_list = [[pred for vk, v_dict in sorted(predictions.items())
                      for fk, f_dict in sorted(v_dict.items())
                      for pk, pred in sorted(f_dict.items()) if pred[8] == idx] for idx in range(4)]

        error_rates = [np.mean([pred[7] for pred in preds]) for preds in pred_list]
        logger.info("Towards {} / Away {} / Across {} / Other {}".format(*error_rates))

        prediction_path = os.path.join(save_dir, "prediction.json")
        with open(prediction_path, "w") as f:
            json.dump(prediction_dict, f)


if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)
    start = time.time()

    logger.initialize(args.root_dir)
    logger.info(vars(args))
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))
    summary = SummaryLogger(args, logger, os.path.join(args.root_dir, "summary.csv"))
    summary.update("finished", 0)

    data_dir = "data"
    data = joblib.load(args.in_data)
    traj_len = data["trajectories"].shape[1]

    # Load training data
    train_splits = list(filter(lambda x: x != args.eval_split, range(args.nb_splits)))
    valid_split = args.eval_split + args.nb_splits

    train_dataset = SceneDatasetCV(data, args.input_len, args.offset_len, args.pred_len,
                                   args.width, args.height, data_dir, train_splits, args.nb_train,
                                   True, "scale" in args.model, args.ego_type)
    logger.info(train_dataset.X.shape)
    valid_dataset = SceneDatasetCV(data, args.input_len, args.offset_len, args.pred_len,
                                   args.width, args.height, data_dir, valid_split, -1,
                                   False, "scale" in args.model, args.ego_type)
    logger.info(valid_dataset.X.shape)

    # X: input, Y: output, poses, egomotions
    # dataset.py __getitem__函数返回7个变量，选其中4个作为模型输入给forward
    data_idxs = [0, 1, 2, 7]
    if data_idxs is None:
        logger.info("Invalid argument: model={}".format(args.model))
        exit(1)

    model = get_model(args).to(args.device)  # 确保get_model返回的是PyTorch模型

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_step_list, gamma=0.5)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nb_jobs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nb_jobs)

    train_eval = Evaluator("train", args)
    valid_eval = Evaluator("valid", args)

    logger.info("Training...")
    train_eval.reset()
    st = time.time()

    for iter_cnt, batch in enumerate(train_loader):
        if iter_cnt == args.nb_iters:
            print("Break!!!")
            break
        model.train()
        model.zero_grad()  # 清除旧的梯度
        # 将数据和标签移动到相应设备
        # inputs = [batch[idx].to(args.device) for idx in data_idxs]
        # inputs = [torch.cat([x[idx].to(args.device) for x in batch], dim=0) for idx in data_idxs]
        inputs = concat_examples_pytorch(batch, args.device, data_idxs)
        print("----------------")
        print(len(inputs))
        print(len(inputs[0]))
        if iter_cnt == 2:
            break
        # 前向传播
        pred_y, pos_y = model(inputs)
        loss = F.mse_loss(pred_y, pos_y)  # 计算损失

        # 反向传播和优化
        loss.backward()  # 计算梯度

        optimizer.step()  # 更新参数
        scheduler.step()  # 更新学习率

        # 更新训练评估
        train_eval.update(loss.item(), pred_y, batch)

        # 验证和报告
        if (iter_cnt + 1) % args.iter_snapshot == 0:
            logger.info("Validation...")
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(save_dir, "model_{}.pth".format(iter_cnt + 1)))
            valid_eval.reset()
            validate_and_report(model, valid_loader, args.device, args, save_dir, iter_cnt)
            logger.info("Validation completed.")
            # 重置时间统计器
            st = time.time()
            train_eval.reset()

        elif (iter_cnt + 1) % args.iter_display == 0:
            print("----------------------")
            print("loss = " + str(loss))
            msg = "Iter {}: train loss {} / ADE {} / FDE {}"
            logger.info(msg.format(iter_cnt + 1, train_eval("loss"), train_eval("ade"), train_eval("fde")))
    summary.update("finished", 1)
    summary.write()
    logger.info("Elapsed time: {} (s), Saved at {}".format(time.time() - start, save_dir))

'''batch - (11,batch_size)
`batch`实际上是一个包含两个或多个元素的元组（或者是列表），而不是直接包含数据项的单一对象。这种情况通常发生在当你的`Dataset`对象返回的每个项是一个元组，比如`(data, label)`，而`DataLoader`将这些项集合成批次时。这时，`len(batch)`实际上给出的是元组中元素的数量，通常对应于数据和标签（如果有其他元素，数字可能更大）。
在这种情况下，`batch[0]`可能是所有数据项的集合，而`batch[1]`是所有标签的集合。因此，`len(batch[0])`实际上是批量大小，也就是每个批次中数据项的数量，这与你设置的`batch_size`相匹配。
举个例子，如果你的数据集返回的每个项是一个形如`(image, label)`的元组，那么经过`DataLoader`处理后，每个`batch`将是形如`([images], [labels])`的结构，其中`[images]`和`[labels]`分别是图片和标签的列表（或张量）。在这种情况下：
- `len(batch)`将返回2（因为`batch`包含两个元素：数据和标签）。
- `len(batch[0])`将返回批次中的数据项数量，即`batch_size`。
这解释了为什么你会观察到`len(batch[0]) = batch_size`，而不是`len(batch) = batch_size`的情况。
'''
