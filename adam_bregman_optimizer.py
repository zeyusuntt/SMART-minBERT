from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamWBreg(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-5,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            beta_bppo: float = 0.1
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)
        self.beta_bppo = beta_bppo

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"] += 1

                # get moments
                m, v = state["exp_avg"], state["exp_avg_sq"]
                # get hyperparameters
                b1, b2 = group["betas"]
                bt1 = b1 ** state["step"]
                bt2 = b2 ** state["step"]

                # update moments inplace
                m.mul_(b1).add_((1 - b1) * grad)
                v.mul_(b2).add_((1 - b2) * grad ** 2)
                # bias correction(efficient version)
                alpha_t = alpha * math.sqrt(1 - bt2) / (1 - bt1)
                update = alpha_t * m / (torch.sqrt(v) + group["eps"])
                # Bregman optimization update
                update += self.beta_bppo * grad
                p.data -= update
                # update again using weight decay
                p.data -= alpha * group["weight_decay"] * p.data
        return loss
