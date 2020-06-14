import torch.nn as nn
import numpy as np
import torch
# import torch.nn.functional as F
from torchvision import models
from im2mesh.common import normalize_imagenet


class ConvEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 512, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out

class Recurrent_BatchNorm3d(nn.Module):
    #use similar APIs as torch.nn.BatchNorm3d.
    def __init__(self, \
                 num_features, \
                 T_max, \
                 eps=1e-5, \
                 momentum=0.1, \
                 affine=True, \
                 track_running_stats=True):
        super(Recurrent_BatchNorm3d, self).__init__()
        #num_features is C from an expected input of size (N, C, D, H, W)
        self.num_features = num_features
        self.T_max = T_max
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats=True

        #if affine is true, this module has learnable affine parameters
        #weight is gamma and bias is beta in the batch normalization formula
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        #if track_running_stats is True, this module will track the running mean and running variance
        for i in range(T_max):
            self.register_buffer('running_mean_{}'.format(i), \
                                 torch.zeros(num_features) if track_running_stats else None)
            self.register_buffer('running_var_{}'.format(i), \
                                 torch.zeros(num_features) if track_running_stats else None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            for i in range(self.T_max):
                running_mean = getattr(self, 'running_mean_{}'.format(i))
                running_var = getattr(self, 'running_var_{}'.format(i))

                running_mean.zero_()
                running_var.fill_(1)

        if self.affine:
            #according to the paper, 0.1 is a good initialization for gamma
            self.weight.data.fill_(0.1)
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
            if input_.dim() != 5:
                raise ValueError('expected 5D input (got {}D input)'.format(input_.dim()))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.T_max:
            time = self.T_max - 1

        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))

        return nn.functional.batch_norm(input_, \
                                        running_mean = running_mean, \
                                        running_var = running_var, \
                                        weight = self.weight, \
                                        bias = self.bias, \
                                        training = self.training, \
                                        momentum = self.momentum, \
                                        eps = self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' T_max={T_max}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

class BN_FCConv3DLayer_torch(nn.Module):
    def __init__(self, fc_w_fan_in, filter_shape, output_shape, n_views=24):
        print("initializing \"FCConv3DLayer_torch\"")
        super(BN_FCConv3DLayer_torch, self).__init__()
        self.output_shape = output_shape

        #fc_layer is not the same as fc7
        self.fc_layer = nn.Linear(fc_w_fan_in, int(np.prod(output_shape[1:])), bias=False)

        #filter_shape = (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
        self.conv3d = nn.Conv3d(filter_shape[0], filter_shape[1], \
                                kernel_size= filter_shape[2], \
                                padding= int((filter_shape[2] - 1) / 2), bias=False)

        #define the recurrent batch normalization layers
        #input channels is the output channels of FCConv3DLayer_torch and T_max is the maximum number of views
        self.bn1 = Recurrent_BatchNorm3d(num_features = filter_shape[0], T_max = n_views)
        self.bn2 = Recurrent_BatchNorm3d(num_features = filter_shape[0], T_max = n_views)

        #define a bias term and initialize it to 0.1
        self.bias = nn.Parameter(torch.FloatTensor(1, output_shape[1], 1, 1, 1).fill_(0.1))

    def forward(self, fc7, h, time):
        #fc7 is the leakyReLU-ed ouput of fc7 layer
        #h is the hidden state of the previous time step
        target_shape = list(self.output_shape)

        # To deal with different batch_size.
        target_shape[0] = -1

        fc7 = self.fc_layer(fc7).view(*target_shape)
        bn_fc7 = self.bn1(fc7, time)    #the input of Recurrent_BatchNorm3d is (input_, time)

        conv3d = self.conv3d(h)
        bn_conv3d = self.bn2(conv3d, time)

        out = bn_fc7 + bn_conv3d + self.bias
        return out

class ResidualGRU18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    '''

    def __init__(self, c_dim, batch_size, fc_size, n_convilter, n_deconvfilter, n_gru_vox, conv3d_filter_shape, h_shape, normalize=True, n_views=24):
        super().__init__()
        self.normalize = normalize
        self.batch_size = batch_size
        #self.use_linear = use_linear
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()

        self.leaky_relu = nn.LeakyReLU(negative_slope= 0.01)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc = nn.Linear(512, fc_size)
        #define the FCConv3DLayers in 3d convolutional gru unit
        #conv3d_filter_shape = (self.n_deconvfilter[0], self.n_deconvfilter[0], 3, 3, 3)
        self.t_x_s_update = BN_FCConv3DLayer_torch(fc_size, conv3d_filter_shape, h_shape, n_views)
        self.t_x_s_reset = BN_FCConv3DLayer_torch(fc_size, conv3d_filter_shape, h_shape, n_views)
        self.t_x_rs = BN_FCConv3DLayer_torch(fc_size, conv3d_filter_shape, h_shape, n_views)

    def forward(self, x, h, u, time):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        fc = self.fc(net)

        rect = self.leaky_relu(fc)

        t_x_s_update = self.t_x_s_update(rect, h, time)
        t_x_s_reset = self.t_x_s_reset(rect, h, time)

        update_gate = self.sigmoid(t_x_s_update)
        complement_update_gate = 1 - update_gate
        reset_gate = self.sigmoid(t_x_s_reset)

        rs = reset_gate * h
        t_x_rs = self.t_x_rs(rect, rs, time)
        tanh_t_x_rs = self.tanh(t_x_rs)

        gru_out = update_gate * h + complement_update_gate * tanh_t_x_rs
        return gru_out, update_gate


class Resnet34(nn.Module):
    r''' ResNet-34 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet50(nn.Module):
    r''' ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet101(nn.Module):
    r''' ResNet-101 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out
