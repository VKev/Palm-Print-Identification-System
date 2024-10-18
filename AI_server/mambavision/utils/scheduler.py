import torch
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr=1e-3,
        min_lr=1e-5,
        warmup_epochs=5,
        plateau_epochs=5,
        total_epochs=100,
    ):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.plateau_epochs = plateau_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch <= self.warmup_epochs:
            # Linear decay from initial_lr to 1e-4
            return [
                self.initial_lr
                - (self.initial_lr - 1e-4) * (epoch / self.warmup_epochs)
                for _ in self.base_lrs
            ]
        elif epoch <= self.warmup_epochs + self.plateau_epochs:
            # Plateau at 1e-4
            return [1e-4 for _ in self.base_lrs]
        else:
            # Linear decay from 1e-4 to min_lr
            remaining_epochs = self.total_epochs - (
                self.warmup_epochs + self.plateau_epochs
            )
            current_epoch = epoch - (self.warmup_epochs + self.plateau_epochs)
            return [
                1e-4 - (1e-4 - self.min_lr) * (current_epoch / remaining_epochs)
                for _ in self.base_lrs
            ]

    def step(self):
        self.last_epoch += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
