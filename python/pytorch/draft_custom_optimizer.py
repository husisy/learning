import torch


# see https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html

# https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
class CustomSGD(torch.optim.Optimizer):

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0):
        assert lr > 0
        assert momentum >= 0
        assert weight_decay >= 0
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay > 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        d_p = param_state['momentum_buffer']
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1-dampening)
                        d_p = param_state['momentum_buffer']

                p.add_(d_p, alpha=-group['lr'])

        return loss


# see https://pytorch.org/docs/stable/_modules/torch/optim/adam.html
class CustomAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        assert lr > 0
        assert eps > 0
        assert 0 <= betas[0] < 1
        assert 0 <= betas[1] < 1
        assert weight_decay >= 0
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                if group['weight_decay'] > 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(1-beta2**state['step'])).add_(group['eps'])

                p.addcdiv_(exp_avg, denom, value=-group['lr']/(1-beta1**state['step']))

        return loss
