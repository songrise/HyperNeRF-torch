# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#%%
"""Modules for NeRF models."""

import functools
from typing import Any, Optional, Tuple


import torchsummary as summary #testing
import torch
import torch.nn as nn
import numpy as np

from hypernerf import model_utils
# from hypernerf import types

def get_norm_layer(norm_type:str,channels:int):
    """Returns a norm layer.
    Args:
        norm_type: A string indicating the type of norm layer to return.
    Returns:
        A norm layer.
    """
    if norm_type == 'batch':
        return nn.BatchNorm2d(num_features=channels)
    elif norm_type == 'instance':
        return nn.InstanceNorm2d(num_features=channels)
    else:
        raise ValueError('Unknown norm type: {}'.format(norm_type))
        return None

class Dense(nn.Module):
    """A dense layer."""

    def __init__(self, in_channels: int, out_channels: int,
                 activation: Optional[str] = None,
                 norm: Optional[str] = None,
                 dropout: Optional[float] = None):
        """Initializes a dense layer.
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            activation: A string indicating the activation function.
            norm: A string indicating the type of normalization.
            dropout: The dropout rate.
        """
        super(Dense, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm = norm
        self.dropout = dropout
        self.dropout_layer = None
        self.norm_layer = None
        self.activation_layer = None
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)
        if self.norm is not None:
            self.norm_layer = get_norm_layer(self.norm)
        if self.activation is not None:
            self.activation_layer = model_utils.get_activation_layer(self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: A tensor of shape (..., in_channels).
        Returns:
            A tensor of shape (..., out_channels).
        """
        x = torch.reshape(x, (-1, self.in_channels))
        x = torch.matmul(x, self.weight)

        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.activation_layer is not None:
            x = self.activation_layer(x)


class MLP(nn.Module):
    """A multi-layer perceptron.
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        hidden_channels: The number of hidden channels.
        hidden_layers: The number of hidden layers.
        hidden_activation: The activation function for the hidden layers.
        hidden_norm: A string indicating the type of norm to use for the hidden
            layers.
        out_activation: The activation function for the output layer.
        out_norm: A string indicating the type of norm to use for the output
            layer.
        dropout: The dropout rate.
    """

    def __init__(self,in_ch,out_ch, depth=8,width=256,hidden_init=None,hidden_activation=None,
            hidden_norm=None,output_init=None,output_channels=0,output_activation=None,
            use_bias=True,skips=None):
        super(MLP, self).__init__() 
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.depth = depth
        self.width = width
        if hidden_init is None:
            self.hidden_init = nn.init.kaiming_normal_
        else:
            self.hidden_init = hidden_init

        if hidden_activation == None:
            self.hidden_activations = nn.ReLU()
        else:
            self.hidden_activations = hidden_activation

        self.hidden_norm = hidden_norm

        if output_init is None:
            self.output_init = nn.init.kaiming_normal_
        else:
            self.output_init = output_init

        self.output_channels = output_channels
        if output_activation == None:
            self.output_activation = nn.Identity()
        else:
            self.output_activation = output_activation

        self.use_bias = use_bias
        if skips is None:
            self.skips = [4,]
        else:
            self.skips = skips

        self.linears = nn.ModuleList([nn.Linear(in_ch, width)] + 
            [nn.Linear(width, width) if i not in self.skips else 
            nn.Linear(width+ in_ch, width) for i in range(depth-1)])
        self.logit_layer = nn.Linear(width, out_ch)

        # initalize using glorot
        for _, linear in enumerate(self.linears):
            self.hidden_init(linear.weight)
        # initialize output layer
        if self.output_init is not None:
            self.output_init(self.logit_layer.weight)

        if self.hidden_norm is not None:
            #TODO
            pass
            # self.norm_layers = nn.ModuleList([get_norm_layer(self.hidden_norm) for _ in range(depth)])

    def forward(self,inputs):
        x = inputs
        for i, linear in enumerate(self.linears):
            x = linear(x)
            x = self.hidden_activations(x)
            # if self.hidden_norm is not None:
            #     x = self.norm_layers[i](x)
            if i in self.skips:
                x = torch.cat([x,inputs],-1)
        x = self.logit_layer(x)
        x = self.output_activation(x)
        return x



class GLOEmbed(nn.Module):
    """A GLO encoder module, which is just a thin wrapper around nn.Embed.

    Attributes:
        num_embeddings: The number of embeddings.
        features: The dimensions of each embedding.
        embedding_init: The initializer to use for each.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, embedding_init = None):
        super(GLOEmbed, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if embedding_init is None:
            # todo temp not used
            pass
            # self.embedding_init = nn.init.kaiming_normal_
        self.embedding_init = embedding_init
        self.embed = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Method to get embeddings for specified indices.

        Args:
        inputs: The indices to fetch embeddings for.

        Returns:
        The embeddings corresponding to the indices provided.
        """
        if inputs.shape[-1] == 1:
            inputs = torch.squeeze(inputs, axis=-1)

        return self.embed(inputs)



    
class NerfMLP(nn.Module):
    """A simple MLP.

    Attributes:
        nerf_trunk_depth: int, the depth of the first part of MLP.
        nerf_trunk_width: int, the width of the first part of MLP.
        nerf_rgb_branch_depth: int, the depth of the second part of MLP.
        nerf_rgb_branch_width: int, the width of the second part of MLP.
        activation: function, the activation function used in the MLP.
        skips: which layers to add skip layers to.
        alpha_channels: int, the number of alpha_channelss.
        rgb_channels: int, the number of rgb_channelss.
        condition_density: if True put the condition at the begining which
        conditions the density of the field.
    """

    def __init__(self,in_ch,out_ch=3,trunk_depth=8, trunk_width=256, 
                rgb_branch_depth=1,rgb_branch_width=128,rgb_channels=3,
                alpha_brach_depth=1,alpha_brach_width=128,alpha_channels=1,
                skips=None,hidden_activation=None,rgb_activation= None,
                sigma_activation=None, norm=None):
        super(NerfMLP, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch#todo remove
        self.trunk_depth = trunk_depth
        self.trunk_width = trunk_width
        self.rgb_branch_depth = rgb_branch_depth
        self.rgb_branch_width = rgb_branch_width
        self.rgb_channels = rgb_channels
        self.alpha_branch_depth = alpha_brach_depth
        self.alpha_branch_width = alpha_brach_width
        self.alpha_channels = alpha_channels
        self.condition_density = False
        if skips is None:
            self.skips = [4,]
        self.skips = skips
        if hidden_activation == None:
            self.hidden_activation = nn.ReLU()
        self.hidden_activation = hidden_activation
        if rgb_activation == None:
            self.rgb_activation = nn.Identity()
        self.rgb_activation = rgb_activation
        if sigma_activation == None:
            self.sigma_activation = nn.Identity()
        self.sigma_activation = sigma_activation
        self.norm = norm

        #todo check this
        self.trunk_mlp = MLP(in_ch=self.in_ch,out_ch=self.trunk_width,depth=self.trunk_depth,
            width=self.trunk_width)

        self.bottleneck_mlp = nn.Linear(self.trunk_width,self.trunk_width//2)#128

        #! Jun 26: x2 for concat the condition
        # todo check in dimension
        # todo assume have rgb conditioning, HARDCODED!
        self.rgb_mlp = MLP(in_ch=self.rgb_branch_width+24,
                            out_ch=self.rgb_channels,
                            depth=self.rgb_branch_depth,
                            hidden_activation=self.hidden_activation,
                            output_activation=self.rgb_activation, 
                            width=self.rgb_branch_width,
                            skips=self.skips)
        #todo assume no alpha conditioning (256)
        self.alpha_mlp = MLP(in_ch=self.trunk_width,
                                out_ch=self.alpha_channels,
                                depth=self.alpha_branch_depth,
                                hidden_activation=self.hidden_activation, 
                                output_activation=self.sigma_activation,
                                width=self.alpha_branch_width,
                                skips=self.skips,
                                )
        
    def broadcast_condition(self,c,num_samples):
        # Broadcast condition from [batch, feature] to
        # [batch, num_coarse_samples, feature] since all the samples along the
        # same ray has the same viewdir.
        if c.dim() == 2:
            c = c.unsqueeze(1)
        c = c.repeat(1,num_samples,1)        
        # Collapse the [batch, num_coarse_samples, feature] tensor to
        # [batch * num_coarse_samples, feature] to be fed into nn.Dense.
        # c = c.view([-1, c.shape[-1]])
        return c

    def forward(self,x, alpha_condition = None,rgb_condition= None):
        """
            Args:
            x: sample points with shape [batch, num_coarse_samples, feature].
            alpha_condition: a condition array provided to the alpha branch.
            rgb_condition: a condition array provided in the RGB branch.

            Returns:
            raw: [batch, num_coarse_samples, rgb_channels+alpha_channels].
        """
        x = self.trunk_mlp(x)
        bottleneck = self.bottleneck_mlp(x)
        if alpha_condition is not None:
            alpha_condition = self.broadcast_condition(alpha_condition,x.shape[1])
            alpha_input = torch.cat([bottleneck,alpha_condition],dim=-1)
        else:
            alpha_input = x
        # todo when assuming no alpha conditioning,
        # the input to alpha_mlp should be the bottleneck,ie 256
        alpha = self.alpha_mlp(alpha_input)

        if rgb_condition is not None:
            rgb_condition = self.broadcast_condition(rgb_condition,x.shape[1])
            rgb_input = torch.cat([bottleneck,rgb_condition],dim=-1)
        else:
            rgb_input = x
        rgb = self.rgb_mlp(rgb_input)
        #todo reshape?
        return {'rgb':rgb,'alpha':alpha}



class HyperSheetMLP(nn.Module):
    def __init__(self,in_ch=3,in_ch_embed=8,out_ch=3,depth=6,width=64,min_deg=0,max_deg =1,skips=None):
        super(HyperSheetMLP, self).__init__()
        self.out_ch = out_ch
        self.depth = depth
        self.width = width
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.in_ch_embed = in_ch_embed # default is 8 according to the paper
        self.in_ch = model_utils.get_posenc_ch(in_ch,self.min_deg,self.max_deg,alpha=None) + in_ch_embed
        if skips is None:
            self.skips = [4,]
        else:
            self.skips = skips
        self.hidden_init = nn.init.xavier_normal_
        self.output_init = functools.partial(nn.init.normal_,std=1e-5)
        self.mlp = MLP(in_ch=self.in_ch,out_ch=self.out_ch,
        depth=self.depth,hidden_init = self.hidden_init, 
        output_init = self.output_init,width=self.width,
        skips=self.skips)


    def forward(self,pts,embed,alpha = None):
        points_feat = model_utils.posenc(pts,self.min_deg,self.max_deg,alpha=alpha)#todo
        inputs = torch.cat([points_feat,embed],dim=-1)
        return self.mlp(inputs)

if __name__ == "__main__":
    # test MLP
    device = torch.device('cuda:0')
    mlp = MLP(in_ch=3,out_ch=3,depth=8,width=256).to(device)
    x = torch.randn(1,3,3).cuda()
        

    print(mlp(x))
    # test NerfMLP
    nerf_mlp = NerfMLP(in_ch=3,out_ch=3,trunk_depth=8, trunk_width=256, 
                gb_branch_depth=1,rgb_branch_width=128,rgb_channels=3,
                alpha_brach_depth=0,alpha_brach_width=128,alpha_channels=1,skips=None).to(device)
    x = torch.randn(10,3,3).cuda()
    # (Batch, feature), (1,256)
    alpha_condition = torch.rand(10,256).cuda()
    rgb_condition = torch.randn(10,256).cuda()
    print(nerf_mlp(x,alpha_condition,rgb_condition))

    # test HyperSheetMLP

    hyper_mlp = HyperSheetMLP(in_ch=3,in_ch_embed = 21,out_ch=3,depth=6,width=64)
    hyper_mlp = hyper_mlp.to(device)
    x = torch.randn(1,3,3).cuda()
    embed = torch.randn(1,3,21).cuda()
    print(hyper_mlp(x,embed))
    # summary.summary(hyper_mlp,[(1,3,3),(1,3,21)])
    summary.summary(nerf_mlp,[(1,3,3),(1,1,256),(1,1,256)])
# %%
