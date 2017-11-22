from torch import nn


class conv2d(nn.Module):
    """Applied a 2D connolution over an input signal composed of several input planes.
    
    This method based on torch.nn.Conv2d, learning from mxnet/gluon and tensorflow/slim.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution
        padding (int or tuple, optional): Zero-padding added to both sides of the input
        bias (bool, optional): If True, adds a learnable bias to the output       
        dilation (int or tuple, optional): Spacing between kernel elements
        groups (int, optional): Number of blocked connections from input channels to output channels
        activation(str or function, optional): Activation function used, accept str or certain activation function, like 'relu' or nn.ReLU(), default is None
        normalizer(bool, optional): If True, BatchNormalization will be used, default is False
        normalizer_param(dict, optional): Batch Normalization parameters dictionary, 'e' means eps, 'm' means momentum, 'a' means affine
        weight_initializer(function, optional): Customer weight initializer function applying to convolutional weight
        bias_initializer(function, optional): Customer bias initializer applying to convolutional bias
        order(str, optional): Convolution, activation function and batch normalization order, default is Convolution -> Batch Normalization -> Activation 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 dilation=1,
                 groups=1,
                 activation=None,
                 normalizer=False,
                 normalizer_param={'e': 1e-5,
                                   'm': 0.1,
                                   'a': True},
                 weight_initializer=None,
                 bias_initializer=None,
                 order='CNA'):
        super(conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.order = order

        activation_dict = {
            'relu': nn.ReLU(True),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax2d(),
            'logsigmoid': nn.LogSigmoid()
        }
        if isinstance(activation, str):
            activation_fn = activation_dict[activation]
        else:
            activation_fn = activation

        if normalizer:
            normalizer_fn = nn.BatchNorm2d(out_channels, normalizer_param['e'],
                                           normalizer_param['m'],
                                           normalizer_param['a'])
        else:
            normalizer_fn = None

        assert order in ['CNA', 'CAN', 'ACN', 'ANC', 'NCA',
                         'NAC'], 'Conv, BN and Activation order is illegal'
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding, dilation, groups, bias)
        if weight_initializer is not None:
            weight_initializer(conv_layer.weight)
        if bias_initializer is not None:
            bias_initializer(conv_layer.bias)

        self.weight = conv_layer.weight
        self.bias = conv_layer.bias

        layer_dict = {'C': conv_layer, 'A': activation_fn, 'N': normalizer_fn}
        # get the order sequential
        self.layer = nn.Sequential()
        for i in order:
            if layer_dict[i] is not None:
                self.layer.add_module(i, layer_dict[i])

    def forward(self, x):
        return self.layer(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
