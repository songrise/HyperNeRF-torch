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

# pylint: disable=invalid-name
# pytype: disable=attribute-error

#%%
import torch

def matmul(a,b):
    return torch.matmul(a,b)

def skew(w):
    """Build a skew matrix ("cross product matrix") for vector w.

    Modern Robotics Eqn 3.30.

    Args:
        w: (3,) A 3-vector

    Returns:
        W: (3, 3) A skew matrix such that W @ v == w x v
    """
    w = w.view(3)
    return torch.tensor([[0, -w[2], w[1]],
                        [w[2], 0.0, -w[0]],
                        [-w[1], w[0], 0.0]], dtype=torch.float32).cuda()

def rp_to_se3(r, p):
    """Build a SE3 matrix from a rotation matrix and a translation vector.

    Args:
        r: (3, 3) A rotation matrix
        p: (3,) A translation vector

    Returns:
        T: (4, 4) A homogeneous transformation matrix
    """
    p = p.view(3,1)
    up =  torch.cat([r, p], dim=1)
    lower = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()
    return torch.cat([up, lower], dim=0)

def exp_so3(w, theta):
    w = skew(w)
    return torch.eye(3).cuda() + torch.sin(theta) * w + (1.0-torch.cos(theta)) * (w @ w)

def exp_se3(S, theta):
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.88.

    Args:
        S: (,6) A screw axis of motion.
        theta: Magnitude of motion.

    Returns:
        a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    w,v = S[...,:3], S[...,3:]
    v = v.squeeze(0)#todo check
    W = skew(w)
    R = exp_so3(w, theta).cuda()
    a = (theta * torch.eye(3).cuda() + (1.0 - torch.cos(theta)) * W +
              (theta - torch.sin(theta)) * matmul(W, W))

    # reshape v to match the shape
    v = v.view(v.shape[1], v.shape[0])
    p = matmul((theta * torch.eye(3).cuda() + (1.0 - torch.cos(theta)) * W +
              (theta - torch.sin(theta)) * matmul(W, W)), v)
    return rp_to_se3(R, p)

def to_homogenous(v):
    ones = torch.ones_like(v[..., :1]).cuda()
    #convert from (N,1,4) to (4,N)
    res = torch.cat([v, ones], dim=-1).squeeze(0)
    return res.reshape(res.shape[1], res.shape[0])

def from_homogenous(v):

    return v[..., :3] / v[..., -1:]

if __name__ == '__main__':
    screw_axis = torch.ones(1,1,6).cuda()
    theta = torch.ones(1,1).cuda()
    transform = exp_se3(screw_axis, theta)