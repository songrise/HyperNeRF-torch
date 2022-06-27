import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def posenc(x, min_deg, max_deg, use_identity=False, alpha=None):
    """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1]."""
    batch_shape = x.shape[:-1]
    scales = 2.**torch.linspace(min_deg, max_deg, steps=max_deg-min_deg).cuda()
    # (*, F, C).
    xb = x[..., None, :] * scales[:, None]
    # (*, F, 2, C).
    four_feat = torch.sin(torch.stack((xb, xb + 0.5*3.1415926),dim = -2)).cuda()
    
    if alpha is not None:
        window = posenc_window(min_deg, max_deg, alpha)
        four_feat = window[..., None, None] * four_feat

    # (*, 2*F*C).
    four_feat = four_feat.view((*batch_shape, -1))

    if use_identity:
        return torch.cat([x, four_feat], dim=-1)
    else:
        return four_feat

def get_posenc_ch(in_ch, min_deg, max_deg, use_identity=False, alpha=None):
    """get the channels of the posenc."""
    temp = torch.ones(1,1,in_ch).cuda()
    enc = posenc(temp, min_deg, max_deg, use_identity, alpha)
    return enc.shape[-1]

def posenc_window(min_deg, max_deg, alpha):
    """Windows a posenc using a cosiney window.

    This is equivalent to taking a truncated Hann window and sliding it to the
    right along the frequency spectrum.

    Args:
        min_deg: the lower frequency band.
        max_deg: the upper frequency band.
        alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

    Returns:
        A 1-d numpy array with num_sample elements containing the window.
    """
    bands = torch.linspace(min_deg, max_deg, steps=max_deg-min_deg).cuda()
    x = torch.clamp(alpha - bands, 0.0, 1.0)
    return 0.5 * (1 + torch.cos(3.1416926 * x + 3.1416926))