import os
import json
import time
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.generic import get_args, get_model, write_prediction
from utils.dataset import SceneDatasetCV
from utils.summary_logger import SummaryLogger
from utils.evaluation import Evaluator
from mllogger import MLLogger

logger = MLLogger(init=False)


def concat_examples_pytorch(batch, device, data_idxs):
    batch_array = []
    # batch 11*batch_size
    for idx in data_idxs:
        batch_array.append(batch[idx].to(device))
    return batch_array


if __name__ == "__main__":
    """
    Evaluation with Cross-Validation
    """
    args = get_args()

    torch.manual_seed(args.seed)
    start = time.time()

    logger.initialize(args.root_dir)
    logger.info(vars(args))
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))
    summary = SummaryLogger(args, logger, os.path.join(args.root_dir, "summary.csv"))
    summary.update("finished", 0)

    data_dir = os.getenv("TRAJ_DATA_DIR")
    data = joblib.load(args.in_data)
    traj_len = data["trajectories"].shape[1]

    # Load evaluation data
    valid_split = args.eval_split + args.nb_splits
    valid_dataset = SceneDatasetCV(data, args.input_len, args.offset_len, args.pred_len,
                                   args.width, args.height, data_dir, valid_split, -1,
                                   False, "scale" in args.model, args.ego_type)
    logger.info(valid_dataset.X.shape)

    model = get_model(args)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nb_jobs)
    valid_eval = Evaluator("valid", args)

    # X: input, Y: output, poses, egomotions
    data_idxs = [0, 1, 2, 7]
    if data_idxs is None:
        logger.info("Invalid argument: model={}".format(args.model))
        exit(1)

    logger.info("Evaluation...")

    prediction_dict = {
        "arguments": vars(args),
        "predictions": {}
    }
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在评估时不计算梯度
        # 实现验证逻辑
        for batch in valid_loader:
            inputs = concat_examples_pytorch(batch, args.device, data_idxs)
            loss, pred_y = model(inputs)
            valid_eval.update(loss.item(), pred_y, batch)

            write_prediction(prediction_dict["predictions"], batch, pred_y)

    message_str = "Evaluation: valid loss {} / ADE {} / FDE {}"
    logger.info(message_str.format(valid_eval("loss"), valid_eval("ade"), valid_eval("fde")))
    valid_eval.update_summary(summary, ["loss", "ade", "fde"])
    predictions = prediction_dict["predictions"]
    pred_list = [[pred for vk, v_dict in sorted(predictions.items())
                  for fk, f_dict in sorted(v_dict.items())
                  for pk, pred in sorted(f_dict.items()) if pred[8] == idx] for idx in range(4)]

    error_rates = [np.mean([pred[7] for pred in preds]) for preds in pred_list]
    logger.info("Towards {} / Away {} / Across {} / Other {}".format(*error_rates))

    prediction_path = os.path.join(save_dir, "prediction.json")
    with open(prediction_path, "w") as f:
        json.dump(prediction_dict, f)

    summary.update("finished", 1)
    summary.write()
    logger.info("Elapsed time: {} (s), Saved at {}".format(time.time() - start, save_dir))
