import collections
import logging
import math
import sys
import copy

import torch
import torch.distributed as dist
import functools
import copy

def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def communicate(tensors, communication_op, attention=False):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    flat_tensor = flatten_tensors(tensors)
    communication_op(tensor=flat_tensor)
    if attention:
        return tensors/flat_tensor

    for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
        with torch.no_grad():
            t.set_(f)

# def group_by_dtype(tensors):
#     """
#     Returns a dict mapping from the tensor dtype to a list containing all
#     tensors of that dtype.
#     Arguments:
#         tensors (Iterable[Tensor]): list of tensors.
#     """
#     tensors_by_dtype = collections.defaultdict(list)
#     for tensor in tensors:
#         tensors_by_dtype[tensor.dtype].append(tensor)
#     return tensors_by_dtype
#
# def communicate(tensors, communication_op):
#     """
#     Communicate a list of tensors.
#     Arguments:
#         tensors (Iterable[Tensor]): list of tensors.
#         communication_op: a method or partial object which takes a tensor as
#             input and communicates it. It can be a partial object around
#             something like torch.distributed.all_reduce.
#     """
#     with torch.no_grad():
#         tensors_by_dtype = group_by_dtype(tensors)
#         for dtype in tensors_by_dtype:
#             flat_tensor = flatten_tensors(tensors_by_dtype[dtype])
#             communication_op(tensor=flat_tensor)
#             for f, t in zip(unflatten_tensors(flat_tensor, tensors_by_dtype[dtype]),
#                             tensors_by_dtype[dtype]):
#                 t.set_(f)

def SyncEAvg(model, anchor_model, rank, size, group, alpha):
    '''
    Inputs:
        model: (x^i) local neural net model at i-th worker node
        anchor_model: (z^1=z^2=...=z^m=z) local copy of auxiliary variable
        rank: (i) worker index
        size: (m) total number of workers
        group: worker group
        alpha: (a) elasticity parameter
    Output:
        return void, change in-place
    Formula:
        x_new = (1-a)*x^i + a*z
        z_new = z + a*(sum_i x^i - m*z)
    '''

    for param1, param2 in zip(anchor_model.parameters(), model.parameters()):
        diff = (param2.data - param1.data)
        param2.data = (1-alpha)*param2.data + alpha*param1.data
        param1.data = param1.data/float(size) + alpha*diff

    for param in anchor_model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=group)


def AsyncEAvg(model, anchor_model, rank, size, group, req, alpha):
    '''
    Inputs:
        model: (x^i) local neural net model at i-th worker node
        anchor_model: (z^1=z^2=...=z^m=z) local copy of auxiliary variable
        rank: (i) worker index
        size: (m) total number of workers
        group: worker group
        alpha: (a) elasticity parameter
        req: handle of last iteration's communication
    Output:
        return a handle of asynchronous fuction
    Formula:
        x_new = (1-a)*x^i + a*z
        z_new = z + a*(sum_i x^i - m*z)
        * the computation of z_new isn't finished when the function returns
    '''
    if req:
        for param1, param2 in zip(anchor_model.parameters(), model.parameters()):
            req[param1].wait() # wait the last iteration's update of z to finish

            diff = (param2.data - param1.data)
            param2.data = (1-alpha)*param2.data + alpha*param1.data
            param1.data = param1.data/float(size) + alpha*diff
    else:
        for param1, param2 in zip(anchor_model.parameters(), model.parameters()):
            diff = (param2.data - param1.data)
            param2.data = (1-alpha)*param2.data + alpha*param1.data
            param1.data = param1.data/float(size) + alpha*diff

    for param in anchor_model.parameters():
        req[param] = dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=group, async_op=True)

    return req


def SyncAllreduce(model, rank, size):
    '''
    Inputs:
        model: (x^i) local neural net model at i-th worker node
        anchor_model: (z^1=z^2=...=z^m=z) local copy of auxiliary variable
        rank: (i) worker index
        size: (m) total number of workers
        group: worker group
    Output:
        return void, change in-place
    Formula:
        x_new = sum_i x_i / size
    '''
    communication_op = functools.partial(dist.all_reduce)
    params_list = []
    for param in model.parameters():
        param.data.div_(float(size))

        # params_list.append(param.data)
        params_list.append(param)

    communicate(params_list, communication_op)

def communicate_gather(tensors, rank, gsize, communication_op, group, dst=0, attention=False):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    flat_tensor = flatten_tensors(tensors)
    if rank == 0:
        gather_list = [flat_tensor.clone() for _ in range(gsize)]
    else:
        gather_list = []
    communication_op(tensor=flat_tensor, gather_list=gather_list, group=group, dst=dst)
    if attention:
        return tensors/flat_tensor
    gather_parameters_list = []
    if rank == 0:
        for i in range(gsize):
            # tensors_clone = tensors.clone()
            tensors_clone = copy.deepcopy(tensors)#[ten.clone() for ten in tensors]
            for f, t in zip(unflatten_tensors(gather_list[i], tensors_clone), tensors_clone):
                with torch.no_grad():
                    t.set_(f)

            gather_parameters_list.append(tensors_clone)

        return gather_parameters_list
    else:
        return gather_parameters_list

def SyncAllGather(model, rank, gsize, group):
    '''
    Inputs:
        model: (x^i) local neural net model at i-th worker node
        anchor_model: (z^1=z^2=...=z^m=z) local copy of auxiliary variable
        rank: (i) worker index
        size: (m) total number of workers
        group: worker group
    Output:
        return void, change in-place
    Formula:
        x_new = sum_i x_i / size
    '''
    communication_op = functools.partial(dist.gather)
    params_list = []
    for param in model.parameters():
        params_list.append(param.data.cpu().clone())

    gather_parameters_list = communicate_gather(params_list, rank, gsize, communication_op, group, dst=0)
    return gather_parameters_list


def communicate_1(tensors, communication_op, group, attention=False):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    flat_tensor = flatten_tensors(tensors)
    communication_op(tensor=flat_tensor, group=group)
    if attention:
        return tensors/flat_tensor
    for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
        with torch.no_grad():
            t.set_(f)

def SyncAllreduce_1(model, rank, size,group):
    '''
    Inputs:
        model: (x^i) local neural net model at i-th worker node
        anchor_model: (z^1=z^2=...=z^m=z) local copy of auxiliary variable
        rank: (i) worker index
        size: (m) total number of workers
        group: worker group
    Output:
        return void, change in-place
    Formula:
        x_new = sum_i x_i / size
    '''
    communication_op = functools.partial(dist.all_reduce)
    params_list = []
    for param in model.parameters():
        param.data.div_(float(size))
        # params_list.append(param.data)
        params_list.append(param)

    communicate_1(params_list, communication_op, group=group)

def SyncAllreduce_2(model, rank, size, ue_list):
    '''
    Inputs:
        model: (x^i) local neural net model at i-th worker node
        anchor_model: (z^1=z^2=...=z^m=z) local copy of auxiliary variable
        rank: (i) worker index
        size: (m) total number of workers
        group: worker group
    Output:
        return void, change in-place
    Formula:
        x_new = sum_i x_i / size
    '''
    communication_op = functools.partial(dist.all_reduce)
    params_list = []
    ue_list_set = set(ue_list)
    if rank in ue_list_set:
        for param in model.parameters():
            param.data.div_(float(len(ue_list)))
            # params_list.append(param.data)
            params_list.append(param)
    else:
        for param in model.parameters():
            param.data.mul_(0.0)
            # params_list.append(param.data)
            params_list.append(param)


    communicate(params_list, communication_op)
