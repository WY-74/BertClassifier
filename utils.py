import time
import torch
from torch import nn
from IPython import display
from typing import Tuple, List
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline


def accuracy(y_hat, y) -> float:
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[-1] > 1:
        # 取矩阵每列最大值(每条数据最大概率的类别)，返回一个batchsize长度一维数组作为预测类别
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def init_figsize(nrows: int = 1, ncols: int = 1, figsize: Tuple[int | float, int | float] | None = None):
    if figsize is None:
        figsize = (3.5, 2.5)
    backend_inline.set_matplotlib_formats('svg')
    return plt.subplots(nrows, ncols, figsize=figsize)


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)  # 正确预测的数量，总预测的数量
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(*X), y), y.numel())
    return metric[0] / metric[1]


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[]"""
    device_count = torch.cuda.device_count()
    if device_count == 0:
        return None
    return [torch.device(f'cuda:{i}') for i in range(device_count)]


class Animator:
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Tuple[int | float, int | float] | None = None,
        fmts: Tuple[str, ...] = ("-", "m--", "g-.", "r:"),
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: Tuple[float | None, float | None] = (None, None),
        ylim: Tuple[float | None, float | None] = (None, None),
        xscale: str = "linear",
        yscale: str = "linear",
        legend: List[str] | None = None,
    ):
        self.fig, self.axes = init_figsize(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        display.display(self.fig)
        display.clear_output(wait=True)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def __getitem__(self, idx):
        return self.data[idx]

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
