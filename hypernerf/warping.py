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
"""Warp fields."""
from typing import Any, Iterable, Optional, Dict
from functools import partial
import torch 
import torch.nn as nn

# from hypernerf import model_utils
# from hypernerf import utils
# from hypernerf import modules
# from hypernerf import rigid_body as rigid
# from hypernerf import types
import model_utils
import utils
import modules
import rigid_body as rigid
import types
class TranslationField(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SE3Field(nn.Module):
    """Network that predicts warps as an SE(3) field.

    Attributes:
        points_encoder: the positional encoder for the points.
        metadata_encoder: an encoder for metadata.
        alpha: the alpha for the positional encoding.
        skips: the index of the layers with skip connections.
        depth: the depth of the network excluding the logit layer.
        hidden_channels: the width of the network hidden layers.
        activation: the activation for each layer.
        metadata_encoded: whether the metadata parameter is pre-encoded or not.
        hidden_initializer: the initializer for the hidden layers.
        output_initializer: the initializer for the last logit layer.
    """

    def __init__(self,in_ch=1,out_ch=1):
        super(SE3Field,self).__init__()
        # self.in_ch=in_ch
        self.out_ch=out_ch #not used

        self.min_deg: int = 0
        self.max_deg: int = 8
        self.use_posenc_identity: bool = False

        self.activation = torch.nn.ReLU()
        self.norm: Optional[Any] = None
        self.skips: Iterable[int] = (4,)
        self.trunk_depth: int = 6
        self.trunk_width: int = 128
        self.rotation_depth: int = 0
        self.rotation_width: int = 128
        self.pivot_depth: int = 0
        self.pivot_width: int = 128
        self.translation_depth: int = 0
        self.translation_width: int = 128

        self.default_init  = nn.init.xavier_normal
        self.rotation_init = partial(nn.init.uniform_,b=1e-4)
        self.translation_init = partial(nn.init.uniform_,b=1e-4)
        self.in_ch = model_utils.get_posenc_ch(in_ch,min_deg=self.min_deg,
                                      max_deg=self.max_deg,
                                      use_identity=self.use_posenc_identity,
                                      alpha=None)
        # Unused, here for backwards compatibility.
        num_hyper_dims: int = 0
        hyper_depth: int = 0
        hyper_width: int = 0
        hyper_init = None
        


        self.trunk = modules.MLP(in_ch=self.in_ch,
            out_ch=self.trunk_width,
            depth=self.trunk_depth,
            width=self.trunk_width,
            hidden_activation=self.activation,
            hidden_norm=self.norm,
            hidden_init=self.default_init,
            skips=self.skips)


        self.w_net = modules.MLP(in_ch=self.trunk_width,
                    out_ch=3,#9 for the rotation
                    depth=self.rotation_depth,
                    width=self.rotation_width,
                    hidden_activation=self.activation,
                    hidden_norm=self.norm,
                    hidden_init=self.default_init,
                    output_init=self.rotation_init,
                    output_channels=3)
        self.v_net = modules.MLP(in_ch=self.trunk_width,
                    out_ch=3,#3 for the translation
                    depth=self.translation_depth,
                    width=self.translation_width,
                    hidden_activation=self.activation,
                    hidden_norm=self.norm,
                    hidden_init=self.default_init,
                    output_init=self.translation_init,
                    output_channels=3)
        



    def warp(self,
           points: torch.Tensor,
           metadata_embed: torch.Tensor,
           extra_params: Dict[str, Any]):
        
        points_embed = model_utils.posenc(points,
                                      min_deg=self.min_deg,
                                      max_deg=self.max_deg,
                                      use_identity=self.use_posenc_identity,
                                      alpha=extra_params['warp_alpha'])
        #todo check what is metadata
        # inputs = torch.cat([points_embed, metadata_embed], dim=-1)
        inputs = points_embed

        trunk_output = self.trunk(inputs)

        w = self.w_net(trunk_output)
        v = self.v_net(trunk_output)
        theta = torch.norm(w, dim=-1)
        w = w / theta.unsqueeze(-1)
        v = v / theta.unsqueeze(-1)
        screw_axis = torch.cat([w, v], dim=-1)
        transform = rigid.exp_se3(screw_axis, theta)

        warped_points = points
        warped_points = rigid.from_homogenous(
            torch.matmul(transform, rigid.to_homogenous(warped_points)))

        return warped_points

    def forward(self,
               points: torch.Tensor,
               metadata: torch.Tensor,
               extra_params: Dict[str, Any],
               return_jacobian: bool = False):
        """Warp the given points using a warp field.

        Args:
        points: the points to warp.
        metadata: metadata indices if metadata_encoded is False else pre-encoded
            metadata.
        extra_params: A dictionary containing
            'alpha': the alpha value for the positional encoding.
        return_jacobian: if True compute and return the Jacobian of the warp.

        Returns:
        The warped points and the Jacobian of the warp if `return_jacobian` is
            True.
        """

        out = {
            'warped_points': self.warp(points, metadata, extra_params)
        }

        if return_jacobian:
        # compute the jacobial of the warp
        # jac_fn 
        # jac_fn = jax.jacfwd(self.warp, argnums=0)
        # out['jacobian'] = jac_fn(points, metadata, extra_params)
            raise NotImplementedError
        return out

if __name__ == '__main__':
    inputs= torch.randn(1,1,3)
    inputs = inputs.cuda()
    device = torch.device('cuda')
    model = SE3Field(in_ch=3).to(device)
    res = model(inputs,torch.randn(1,1,3).cuda(),{'warp_alpha':0.5})    
    print(res['warped_points'].shape)
    import torchsummary
    torchsummary.summary(model,[(1,1,3),(1,1,3),{'warp_alpha':0.5}])