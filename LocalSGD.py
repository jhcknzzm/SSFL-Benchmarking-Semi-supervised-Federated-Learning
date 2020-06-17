import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
from comm_helpers import communicate, flatten_tensors, unflatten_tensors
import threading


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, alpha, gmf, size, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0):

        self.alpha = alpha
        self.gmf = gmf
        self.size = size
        self.comm_buf = []



        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)


        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                buf = param_state['anchor_model'] = torch.clone(p.data).detach()
                self.comm_buf.append(buf)

        self.first_flag = True
        self.comm_finish = threading.Event()
        self.buf_ready = threading.Event()
        self.comm_finish.set()
        self.buf_ready.clear()

        self.comm_thread = threading.Thread(
                target=SGD._async_all_reduce_,
                args=(self.comm_buf, self.buf_ready, self.comm_finish))
        self.comm_thread.daemon = True
        self.comm_thread.name = 'Communication-Thread'
        self.comm_thread.start()

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']


            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    def elastic_average(self, itr, cp):
        step_flag = (itr != 0 and itr % cp == 0)
        if step_flag:
            beta = 1/self.size - self.alpha - self.alpha**2/(1-self.alpha)
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    buf = param_state['anchor_model']

                    p.data.mul_(1-self.alpha).add_(self.alpha, buf)
                    buf.mul_(beta).add_(self.alpha/(1-self.alpha), p.data)

            communicate(self.comm_buf, dist.all_reduce)


    def overlap_elastic_average(self, itr, cp, req):
        step_flag = (itr != 0 and itr % cp == 0)
        if step_flag:
            beta = 1/self.size - self.alpha - self.alpha**2/(1-self.alpha)
            gamma = self.alpha/(1-self.alpha)
            if req:
                req.wait()
                for f, t in zip(unflatten_tensors(self.flat_tensor, self.comm_buf), self.comm_buf):
                    t.set_(f)

            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    buf = param_state['anchor_model']

                    p.data.mul_(1-self.alpha).add_(self.alpha, buf)
                    buf.mul_(beta).add_(gamma, p.data)

            self.flat_tensor = flatten_tensors(self.comm_buf)
            req = dist.all_reduce(tensor=self.flat_tensor, async_op=True)

        return req


    def BMUF(self, itr, cp):
        step_flag = (itr != 0 and itr % cp == 0)
        if step_flag:

            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    param_state = self.state[p]
                    old_data = param_state['anchor_model']

                    if 'global_momentum_buffer' not in param_state:
                        buf = param_state['global_momentum_buffer'] = torch.clone(p.data).detach()
                        buf.sub_(old_data)
                        buf.div_(-lr)
                    else:
                        buf = param_state['global_momentum_buffer']
                        buf.mul_(self.gmf).sub_(1/lr, p.data).add_(1/lr, old_data)

                    old_data.add_(-lr, buf)
                    old_data.div_(self.size)

            communicate(self.comm_buf, dist.all_reduce)
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    old_data = param_state['anchor_model']
                    p.data.copy_(old_data)


    def OverlapLocalSGD_step(self, itr, cp, req):
        # Olocal SGD
        step_flag = (itr != 0 and itr % cp == 0)
        if step_flag:

            self.comm_finish.wait()

            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    param_state = self.state[p]
                    old_data = param_state['anchor_model']

                    p.data.mul_(1-self.alpha).add_(self.alpha, old_data)

                    #param_state['momentum_buffer'].zero_()
                    if 'global_momentum_buffer' not in param_state:
                        buf = param_state['global_momentum_buffer'] = torch.clone(p.data).detach()
                        buf.sub_(old_data)
                        buf.div_(-lr)
                    else:
                        buf = param_state['global_momentum_buffer']
                        buf.mul_(self.gmf).sub_(1/lr, p.data).add_(1/lr, old_data)

                    old_data.add_(-lr, buf)
                    old_data.div_(self.size)
                    #param_state['momentum_buffer'].zero_()

            self.comm_finish.clear()
            self.buf_ready.set()

    def async_CoCoD_SGD_step(self, itr, cp, req):
        step_flag = (itr != 0 and itr % cp == 0)
        if step_flag:

            self.comm_finish.wait()

            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    old_data = param_state['anchor_model']

                    if 'local_anchor_model' not in param_state:
                        param_state['local_anchor_model'] = torch.clone(old_data).detach()
                    buf = param_state['local_anchor_model']

                    # update(anchor)
                    old_data.add_(p.data).sub_(buf)

                    # update training params
                    p.data.copy_(old_data)

                    # update local_anchor_model
                    buf.copy_(old_data)

                    old_data.div_(self.size)

            self.comm_finish.clear()
            self.buf_ready.set()


    @staticmethod
    def _async_all_reduce_(buff, buf_ready, comm_finish):
        while True:
            buf_ready.wait()
            communicate(buff, dist.all_reduce)
            buf_ready.clear()
            comm_finish.set()
