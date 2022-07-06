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

"""Quaternion math.

This module assumes the xyzw quaternion format where xyz is the imaginary part
and w is the real part.

Functions in this module support both batched and unbatched quaternions.
"""

import torch

def safe_acos(t,eps=1e-7):
    """Safe arccosine.
    Args:
        t: A tensor of shape (...,1)
        eps: A small number to add to t to avoid division by zero.
    Returns:
        A tensor of shape (...,1)
    """
    return torch.acos(torch.clamp(t, -1.0 + eps, 1.0 - eps))

def im(q):
    """Returns the imaginary part of a quaternion.
    Args:
        q: A tensor of shape (...,4)
    Returns:
        A tensor of shape (...,3)
    """
    return q[...,:3]

def re(q):
    """Returns the real part of a quaternion.
    Args:
        q: A tensor of shape (...,4)
    Returns:
        A tensor of shape (...,1)
    """
    return q[...,3:]
    
def identity():
    """Returns the identity quaternion.
    Returns:
        A tensor of shape (...,4)
    """
    return torch.tensor([0,0,0,1],dtype=torch.float32)

def conjugate(q):
    """Returns the conjugate of a quaternion.
    Args:
        q: A tensor of shape (...,4)
    Returns:
        A tensor of shape (...,4)
    """
    return torch.cat([-im(q),re(q)],dim=-1)

def normalize(q):
    """Returns the normalized quaternion.
    Args:
        q: A tensor of shape (...,4)
    Returns:
        A tensor of shape (...,4)
    """
    return q / torch.norm(q,dim=-1,keepdim=True)

def inverse(q):
    """Returns the inverse of a quaternion.
    Args:
        q: A tensor of shape (...,4)
    Returns:
        A tensor of shape (...,4)
    """
    return normalize(conjugate(q))

def norm(q):
    """Returns the norm of a quaternion.
    Args:
        q: A tensor of shape (...,4)
    Returns:
        A tensor of shape (...,1)
    """
    return torch.norm(q,dim=-1,keepdim=True)

def multiply(q1,q2):
    """Returns the quaternion product of two quaternions.
    Args:
        q1: A tensor of shape (...,4)
        q2: A tensor of shape (...,4)
    Returns:
        A tensor of shape (...,4)
    """
    c = (re(q1) * im(q2)
       + re(q2) * im(q1)
       + torch.cross(im(q1),im(q2)))
    w = re(q1) * re(q2) - torch.dot(im(q1),im(q2))
    return torch.cat([c,w],dim=-1)

def rotate(q,v):
    """Returns the quaternion rotation of a vector.
    Args:
        q: A tensor of shape (...,4)
        v: A tensor of shape (...,3)
    Returns:
        A tensor of shape (...,3)
    """
    q_v = torch.cat([v,torch.zeros_like(v[...,:1])],dim=-1)
    return im(multiply(multiply(q,q_v), conjugate(q)))

def log(q,eps=1e-7):
    """Returns the logarithm of a quaternion.
    Args:
        q: A tensor of shape (...,4)
        eps: A small number to add to q to avoid division by zero.
    Returns:
        A tensor of shape (...,4)
    """
    mag = norm(q)
    v = im(q)
    s = re(q)
    w = torch.log(mag)
    denom = max(norm(v),eps*torch.ones_like(v))
    xyz = v/denom*safe_acos(s/eps)
    return torch.cat([xyz,w],dim=-1)
