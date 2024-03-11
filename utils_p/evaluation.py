import torch


def calc_mse(pred_y, true_y):
    # 使用PyTorch计算MSE，确保操作在Tensor上进行
    mse = torch.mean((pred_y - true_y) ** 2)
    return mse


class Evaluator(object):
    def __init__(self, prefix, args):
        self.prefix = prefix
        self.nb_grids = args.nb_grids
        self.width = args.width
        self.height = args.height
        self.reset()

    def reset(self):
        self.loss = 0
        self.cnt = 0
        self.ade = 0
        self.fde = 0

    def update(self, loss, pred_y, batch):
        batch_size = len(pred_y)
        true_y = batch[1].to(pred_y.device)
        self.loss += loss * batch_size
        mse = calc_mse(pred_y[..., :2], true_y[..., :2])
        self.ade += mse.item() * batch_size  # 转换为Python数值进行累加

        mse = calc_mse(pred_y[:, -1, :2], true_y[:, -1, :2])
        self.fde += mse.item() * batch_size
        self.cnt += batch_size

    def __call__(self, name, normalize=True):
        if normalize:
            return getattr(self, name) / self.cnt if self.cnt != 0 else 0.0
        else:
            return getattr(self, name)

    def update_summary(self, summary, iter_cnt, targets):
        for name in targets:
            summary.update_by_cond(self.prefix + "_" + name, getattr(self, name) / self.cnt, iter_cnt + 1)
        summary.write()
