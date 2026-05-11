import math
import torch
from torch import nn
from torch.optim import lr_scheduler
from utils import set_axes, init_figsize
from utils import accuracy, evaluate_accuracy_gpu
from utils import Timer, Animator, Accumulator


class _Train:
    def __init__(self, net, loss, optimizer, scheduler=None, jit_script: bool = False):
        self.net = self._init_net(net, jit_script)
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _try_all_gpus(self):
        """返回所有可用的GPU，如果没有GPU，则返回[]"""
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return None
        return [torch.device(f'cuda:{i}') for i in range(device_count)]

    def _init_net(self, net, jit_script: bool = False):
        net.apply(self.init_weights)
        if jit_script:
            net = torch.jit.script(net)

        self.devices = self._try_all_gpus()
        self.device = self.devices[0] if self.devices is not None else "cpu"
        if self.devices is not None:
            if len(self.devices) > 1:
                net = net.to(self.device)
                net = nn.DataParallel(net, device_ids=self.devices)
            else:
                net = net.to(self.device)
        else:
            self.devices = ["cpu"]
        return net

    def grad_clipping(self, theta):
        if isinstance(self.net, nn.Module):
            params = [p for p in self.net.parameters() if p.requires_grad]
        else:
            params = self.net.params

        norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    def init_weights(self, m):
        return

    def train_epochs(self, *args, **kwargs):
        return


class _FineTuningBertBase(_Train):
    def train_epochs(self, num_epochs, train_iter, test_iter):
        timer, num_batches = Timer(), len(train_iter)
        animator = Animator(
            xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc']
        )
        for epoch in range(num_epochs):
            self.net.train()
            metric = Accumulator(4)
            for i, (features, labels) in enumerate(train_iter):
                timer.start()
                if isinstance(features, list):
                    X = [x.to(self.device) for x in features]
                else:
                    X = features.to(self.device)
                y = labels.to(self.device)

                self.optimizer.zero_grad()
                pred = self.net(*X)
                l = self.loss(pred, y)
                l.sum().backward()
                self.optimizer.step()

                train_loss_sum = l.sum()
                train_acc_sum = accuracy(pred, y)
                metric.add(train_loss_sum, train_acc_sum, labels.shape[0], labels.numel())
                timer.stop()

                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))
            test_acc = evaluate_accuracy_gpu(self.net, test_iter, self.device)
            animator.add(epoch + 1, (None, None, test_acc))

            if self.scheduler:
                if self.scheduler.__module__ == lr_scheduler.__name__:
                    self.scheduler.step()  # UsingPyTorchIn-Builtscheduler
                else:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.scheduler(epoch, param_group)

        print(f'loss {metric[0] / metric[2]:.3f}, train acc ' f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on ' f'{str(self.devices)}')


class _CosineScheduler:
    def __init__(self, max_update, param_groups, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self._init_lr_orig(param_groups)
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def _init_lr_orig(self, param_groups):
        for param_group in param_groups:
            setattr(self, f"{param_group["name"]}_lr_orig", param_group["lr"])

    def get_warmup_lr(self, epoch, param_group):
        lr_orig = getattr(self, f"{param_group["name"]}_lr_orig")
        increase = (lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def update_lr(self, epoch, param_group):
        lr_orig = getattr(self, f"{param_group["name"]}_lr_orig")
        lr = (
            self.final_lr
            + (lr_orig - self.final_lr) * (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        )
        return lr

    def __call__(self, epoch, param_group):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch, param_group)
        if epoch <= self.max_update:
            setattr(self, f"{param_group["name"]}_lr", self.update_lr(epoch, param_group))
        return getattr(self, f"{param_group["name"]}_lr")


def train(
    net, train_iter, test_iter, num_epochs, max_update, final_lr, warmup_steps, lr={"bert": 2e-5, "outputs": 1e-4}
):
    params_1x = [param for name, param in net.named_parameters() if "bert" in name]
    optimizer = torch.optim.SGD(
        [
            {'name': 'bert', 'params': params_1x, 'lr': lr["bert"]},
            {'name': 'outputs', 'params': net.output.parameters(), "lr": lr["outputs"]},
        ],
        momentum=0.9,
    )
    scheduler = _CosineScheduler(max_update, optimizer.param_groups, final_lr=final_lr, warmup_steps=warmup_steps)
    loss = nn.CrossEntropyLoss(reduction='none')

    _FineTuningBertBase(net, loss, optimizer, scheduler).train_epochs(num_epochs, train_iter, test_iter)

    torch.save(net.state_dict(), "fine-tuning-bert.pth")
