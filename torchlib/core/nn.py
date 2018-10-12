# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Callable, Any, Optional

import torch
import torch.nn.functional as F
from IPython import embed
from torch import Tensor
from torch.nn import *

LambdaFunc = Callable[[Tensor], Tensor]


def range_of(x): return list(range(len(x)))


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


class Lambda(Module):
    "An easy way to create a pytorch layer for a simple `func`."

    def __init__(self, func: LambdaFunc):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)


def ResizeBatch(*size: int) -> Tensor:
    "Layer that resizes x to `size`, good for connecting mismatched layers."
    return Lambda(lambda x: x.view((-1,) + size))


def Flatten() -> Tensor:
    "Flattens `x` to a single dimension, often used at the end of a model."
    return Lambda(lambda x: x.view((x.size(0), -1)))


def PoolFlatten() -> Sequential:
    "Apply `nn.AdaptiveAvgPool2d` to `x` and then flatten the result."
    return Sequential(AdaptiveAvgPool2d(1), Flatten())


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0., actn: Optional[Module] = None):
    "`n_in`->bn->dropout->linear(`n_in`,`n_out`)->`actn`"
    layers = [BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(Dropout(p))
    layers.append(Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


def conv2d(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None, bias=False) -> Conv2d:
    "Create `nn.Conv2d` layer: `ni` inputs, `nf` outputs, `ks` kernel size. `padding` defaults to `k//2`."
    if padding is None:
        padding = ks // 2
    return Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)


def conv_layer(ni: int, nf: int, ks: int = 3, stride: int = 1) -> Sequential:
    "Create Conv2d->BatchNorm2d->LeakyReLu layer: `ni` input, `nf` out filters, `ks` kernel, `stride`:stride."
    return Sequential(
        Conv2d(ni, nf, kernel_size=ks, bias=False,
               stride=stride, padding=ks // 2),
        BatchNorm2d(nf),
        LeakyReLU(negative_slope=0.1, inplace=True))


def conv2d_relu(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None, bn: bool = False,
                bias: bool = False) -> Sequential:
    """Create a `conv2d` layer with `nn.ReLU` activation and optional(`bn`) `nn.BatchNorm2d`: `ni` input, `nf` out
    filters, `ks` kernel, `stride`:stride, `padding`:padding, `bn`: batch normalization."""
    layers = [conv2d(ni, nf, ks=ks, stride=stride,
                     padding=padding, bias=bias), ReLU()]
    if bn:
        layers.append(BatchNorm2d(nf))
    return Sequential(*layers)


def conv2d_trans(ni: int, nf: int, ks: int = 2, stride: int = 2, padding: int = 0) -> ConvTranspose2d:
    """ Create `nn.ConvTranspose2d` layer: `ni` inputs, `nf` outputs, `ks` kernel size,
    `stride`: stride. `padding` defaults to 0."
    :param ni:
    :param nf:
    :param ks:
    :param stride:
    :param padding:
    :return:
    """
    return ConvTranspose2d(ni, nf, kernel_size=ks, stride=stride, padding=padding)


class AdaptiveConcatPool2d(Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz: Optional[int] = None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = AdaptiveAvgPool2d(sz), AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Debugger(Module):
    "A module to debug inside a model."

    def forward(self, x: Tensor) -> Tensor:
        embed()
        return x


class StdUpsample(Module):
    "Increases the dimensionality of our data by applying a transposed convolution layer."

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = conv2d_trans(n_in, n_out)
        self.bn = BatchNorm2d(n_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(F.relu(self.conv(x)))


def std_upsample_head(c, *nfs) -> Module:
    "Create a sequence of upsample layers."
    return Sequential(
        ReLU(),
        *(StdUpsample(nfs[i], nfs[i + 1]) for i in range(4)),
        conv2d_trans(nfs[-1], c)
    )


class CrossEntropyFlat(CrossEntropyLoss):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."

    def forward(self, input: Tensor, target: Tensor):
        n, c, *_ = input.shape
        return super().forward(input.view(n, c, -1), target.view(n, -1))


def simple_cnn(actns, kernel_szs, strides) -> Sequential:
    "CNN with `conv2d_relu` layers defined by `actns`, `kernel_szs` and `strides`."
    nl = len(actns) - 1
    kernel_szs = ifnone(kernel_szs, [3] * nl)
    strides = ifnone(strides, [2] * nl)
    layers = [conv2d_relu(actns[i], actns[i + 1], kernel_szs[i], stride=strides[i])
              for i in range_of(strides)]
    layers.append(PoolFlatten())
    return Sequential(*layers)


def trunc_normal_(x: Tensor, mean: float = 0., std: float = 1.) -> Tensor:
    "Truncated normal initialization."
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


def get_embedding(ni: int, nf: int) -> Module:
    "Create an embedding layer."
    emb = Embedding(ni, nf)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad():
        trunc_normal_(emb.weight, std=0.01)
    return emb
