import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedder import get_embedder


class SDFValidityNet(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        """The network used to estimate the SDF and the validity.
        This implementation of SDF-Net is borrowed from IDR: https://github.com/lioryariv/idr.

        :param d_in: (scalar) The dimension of input channels.
        :param d_out: (scalar) The dimension of output channels.
        :param d_hidden: (scalar) The dimension of hidden layers.
        :param n_layers: (scalar) The number of hidden layers.
        :param skip_in: (tuple) The indices of the hidden layers to concatenate the input.
        :param multires: The maximum frequency of positional embedding.
        :param bias: (scalar) The value to intitalize the bias of the SDF-Net.
        :param scale: (scalar) The scaling of the world positions.
        :param geometric_init: (bool) If use geometric initialization.
        :param weight_norm: (bool) If use weighted normalization in SDF-Net.
        :param inside_outside: (bool) If use negative bias (???)
        """
        super(SDFValidityNet, self).__init__()

        # Implements the SDF-Net.
        # Gets the dimensions of the SDF-Net.
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        # Initializes the positional embedding.
        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        # Creates the SDF-Net.
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        # Implements the Validity-Net.
        self.is_valid_dims = [dims[0], 8, 32, 64, 128, 64, 32, 8, 1]
        for l in range(len(self.is_valid_dims) - 1):
            lin = nn.Linear(self.is_valid_dims[l], self.is_valid_dims[l + 1])
            setattr(self, "is_valid_lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        """Infers the SDF values and validity values.

        :param inputs: (tensor) The world positions, shape = (B, Nsample, 3).
        :return: (tensor) The concatenation of (SDF, Validity, Positional Feature), shape = (N, 1 + 1 + DimFeature).
        """
        batch_size, n_sample, _ = inputs.shape
        inputs = inputs.reshape(-1, 3)
        inputs = inputs * self.scale

        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        # Infers the SDF values and the positional feature.
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)

        # Infers the validity values.
        is_valid = inputs
        for l in range(len(self.is_valid_dims) - 1):
            lin = getattr(self, "is_valid_lin" + str(l))
            is_valid = lin(is_valid)
            if l < len(self.is_valid_dims) - 2:
                is_valid = self.relu(is_valid)
        is_valid = torch.sigmoid(is_valid)

        return torch.cat([x[:, :1] / self.scale, 
                          is_valid,
                          x[:, 1:]], dim=-1)

    def sdf(self, x):
        """Gets the SDF values of the given positions.

        :param x: (tensor) The world positions, shape = (B, Nsample, 3).
        :return: (tensor) The SDF values corresponding to the positions, shape = (B * Nsample, 1)
        """
        return self.forward(x)[:, :1]

    def sdf_withnan(self, x):
        """Gets the SDF values of the given positions, and sets the SDF with low validity to NAN.

        :param x: (tensor) The world positions, shape = (B, Nsample, 3).
        :return: (tensor) The SDF values corresponding to the positions, shape = (B * Nsample, 1)
        """
        vals = self.forward(x).detach()
        sdf = vals[:, :1] / self.scale
        sdf_is_nan = (vals[:, 1:2] * torch.exp(-1000 * sdf * sdf)) < 0.5
        sdf[sdf_is_nan] = float('nan')
        return sdf

    def gradient(self, x):
        """Gets the gradient of the given positions.

        :param x: (tensor) The world positions, shape = (B, Nsample, 3).
        :return: (tensor) The SDF gradients corresponding to the positions, shape = (B * Nsample, 1)
        """
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.reshape(-1, 3)


class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        """The network used to estimate the Color.
        This implementation of Color-Net is borrowed from IDR: https://github.com/lioryariv/idr.

        :param d_feature: (scalar) The dimension of input positional features.
        :param mode: (str) The mode of the Color-Net, choose from:
            (1) idr: Accepts world position, view directions, normals, and positional features as the input. d_in = 9.
            (2) no_view_dir: Accepts world position, normals, and positional features as the input. d_in = 6.
            (3) no_normal: Accepts world position, normals, and positional features as the input. d_in = 6.
            (4) only_pos: Accepts world position and positional features as the input. d_in = 3.
        :param d_in: (scalar) The dimension of input channels.
        :param d_out: (scalar) The dimension of output channels.
        :param d_hidden: (scalar) The dimension of hidden layers.
        :param n_layers: (scalar) The number of hidden layers.
        :param weight_norm: (bool) If use weighted normalization in Color-Net.
        :param multires_view: (int) The maximum frequency of positional embedding.
        :param squeeze_out: (bool) If use Sigmoid at the final output.
        """
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if (self.mode == 'idr' or self.mode == 'no_normal') and multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        """Gets the color of the given positions.

        :param points: (tensor) The world positions, shape = (N, 3).
        :param normals: (tensor) The normal of the world positions, shape = (N, 3).
        :param view_dirs: (tensor) The direction of the ray, shape = (N, 3).
        :param feature_vectors: (tensor) The positional vectors of the world positions, shape = (N, 3).
        :return: (tensor) The color corresponding to the positions, shape = (N, 3)
        """
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        elif self.mode == 'only_pos':
            rendering_input = torch.cat([points, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        """The network used to estimate the variance.
            With high variance values, the volume rendering scheme sets higher rendering weight for the surface region
            and lower rendering weight for the non-surface region.

        This implementation of SingleVarianceNetwork is borrowed from NeuS: https://github.com/Totoro97/NeuS.
        :param init_val: (scalar) The value used to initialize the initial variance.
        """
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        """Gets the variance value.

        :param x: (tensor) The input position, shape = (B, 3). In the NeuS implementation, B = 1.
        :return: (tensor) The variance value, shape = (B, 1).
        """
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
