import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sample_along_rays(device,origins, directions, num_coarse_samples, near, far,
                      use_stratified_sampling, use_linear_disparity):
    """Stratified sampling along the rays.

    Args:
        origins: ray origins.
        directions: ray directions.
        num_coarse_samples: int.
        near: float, near clip.
        far: float, far clip.
        use_stratified_sampling: use stratified sampling.
        use_linear_disparity: sampling linearly in disparity rather than depth.

    Returns:
        z_vals: jnp.ndarray, [batch_size, num_coarse_samples], sampled z values.
        points: jnp.ndarray, [batch_size, num_coarse_samples, 3], sampled points.
    """
    batch_size = origins.shape[0]

    t_vals = torch.linspace(0., 1., num_coarse_samples,device=device)
    if not use_linear_disparity:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    if use_stratified_sampling:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]],dim= -1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand( [batch_size, num_coarse_samples],device=device)
        z_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast z_vals to make the returned shape consistent.
        z_vals = z_vals[None, ...].expand(batch_size, num_coarse_samples)

    return (z_vals, (origins[..., None, :] +
                    z_vals[..., :, None] * directions[..., None, :]))

def volumetric_rendering(device,
                         rgb,
                         sigma,
                         z_vals,
                         dirs,
                         use_white_background,
                         sample_at_infinity=True,
                         eps=1e-10):
    """Volumetric Rendering Function.

    Args:
        rgb: an array of size (B,S,3) containing the RGB color values.
        sigma: an array of size (B,S) containing the densities.
        z_vals: an array of size (B,S) containing the z-coordinate of the samples.
        dirs: an array of size (B,3) containing the directions of rays.
        use_white_background: whether to assume a white background or not.
        sample_at_infinity: if True adds a sample at infinity.
        eps: a small number to prevent numerical issues.

    Returns:
        A dictionary containing:
        rgb: an array of size (B,3) containing the rendered colors.
        depth: an array of size (B,) containing the rendered depth.
        acc: an array of size (B,) containing the accumulated density.
        weights: an array of size (B,S) containing the weight of each sample.
    """
    # TODO(keunhong): remove this hack.
    last_sample_z = 1e10 if sample_at_infinity else 1e-19
    last_sample_z = torch.tensor(last_sample_z, device=device)

    dists = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1],
        last_sample_z.expand(z_vals[..., :1].shape)
    ], -1)

    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * dists)
    # Prepend a 1.0 to make this an 'exclusive' cumprod as in `tf.math.cumprod`.
    accum_prod = torch.cat([
        torch.ones_like(alpha[..., :1], alpha.dtype,device=device),
        torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
    ], dim=-1)
    weights = alpha * accum_prod

    rgb = (weights[..., None] * rgb).sum(dim=-2)
    exp_depth = (weights * z_vals).sum(dim=-1)
    med_depth = compute_depth_map(weights, z_vals)
    acc = weights.sum(dim=-1)
    if use_white_background:
        rgb = rgb + (1. - acc[..., None])

    if sample_at_infinity:
        acc = weights[..., :-1].sum(dim=-1)

    out = {
        'rgb': rgb,
        'depth': exp_depth,
        'med_depth': med_depth,
        'acc': acc,
        'weights': weights,
    }
    return out

def piecewise_constant_pdf(device, bins, weights, num_coarse_samples,
                           use_stratified_sampling):
    """Piecewise-Constant PDF sampling.

    Args:

        bins: jnp.ndarray(float32), [batch_size, n_bins + 1].
        weights: jnp.ndarray(float32), [batch_size, n_bins].
        num_coarse_samples: int, the number of samples.
        use_stratified_sampling: bool, use use_stratified_sampling samples.

    Returns:
        z_samples: jnp.ndarray(float32), [batch_size, num_coarse_samples].
    """
    eps = 1e-5

    # Get pdf
    weights += eps  # prevent nans
    pdf = weights / weights.sum(dim=-1, keepdims=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1],device=device), cdf], axis=-1)

    # Take uniform samples
    if use_stratified_sampling:
        u = torch.rand(list(cdf.shape[:-1]) + [num_coarse_samples],device=device)
    else:
        u = torch.linspace(0., 1., num_coarse_samples,device= device)
        new_shape =  cdf.shape[:-1] + [num_coarse_samples]
        u = u.expand(*new_shape)
    # Invert CDF. This takes advantage of the fact that `bins` is sorted.
    mask = (u[..., None, :] >= cdf[..., :, None])

    def minmax(x):
        #todo check
        x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        x0 = torch.minimum(x0, x[..., -2:-1])
        x1 = torch.maximum(x1, x[..., 1:2])
        return x0, x1

    bins_g0, bins_g1 = minmax(bins)
    cdf_g0, cdf_g1 = minmax(cdf)

    denom = (cdf_g1 - cdf_g0)
    denom = torch.where(denom < eps, 1., denom)
    t = (u - cdf_g0) / denom
    z_samples = bins_g0 + t * (bins_g1 - bins_g0)

    # Prevent gradient from backprop-ing through samples
    return z_samples.detach()
    
def sample_pdf(device, bins, weights, origins, directions, z_vals,
               num_coarse_samples, use_stratified_sampling):
    """Hierarchical sampling.

    Args:

        bins: jnp.ndarray(float32), [batch_size, n_bins + 1].
        weights: jnp.ndarray(float32), [batch_size, n_bins].
        origins: ray origins.
        directions: ray directions.
        z_vals: jnp.ndarray(float32), [batch_size, n_coarse_samples].
        num_coarse_samples: int, the number of samples.
        use_stratified_sampling: bool, use use_stratified_sampling samples.

    Returns:
        z_vals: jnp.ndarray(float32),
        [batch_size, n_coarse_samples + num_fine_samples].
        points: jnp.ndarray(float32),
        [batch_size, n_coarse_samples + num_fine_samples, 3].
    """
    z_samples = piecewise_constant_pdf(device, bins, weights, num_coarse_samples,
                                        use_stratified_sampling)
    # Compute united z_vals and sample points
    z_vals = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
    return z_vals, (
        origins[..., None, :] + z_vals[..., None] * directions[..., None, :])

# Positional encoding (section 5.1), this is the original implementation for NeRF-torch
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

def noise_regularize(raw, noise_std, use_stratified_sampling):
    """Regularize the density prediction by adding gaussian noise.

    Args:

        raw: jnp.ndarray(float32), [batch_size, num_coarse_samples, 4].
        noise_std: float, std dev of noise added to regularize sigma output.
        use_stratified_sampling: add noise only if use_stratified_sampling is True.

    Returns:
        raw: jnp.ndarray(float32), [batch_size, num_coarse_samples, 4], updated raw.
    """
    if (noise_std is not None) and noise_std > 0.0 and use_stratified_sampling:
        noise = torch.rand(raw[..., 3:4].shape, dtype=raw.dtype) * noise_std
        raw = torch.cat([raw[..., :3], raw[..., 3:4] + noise], dim=-1)
    return raw

def compute_opaqueness_mask(device,weights, depth_threshold=0.5):
    """Computes a mask which will be 1.0 at the depth point.

    Args:
        weights: the density weights from NeRF.
        depth_threshold: the accumulation threshold which will be used as the depth
        termination point.

    Returns:
        A tensor containing a mask with the same size as weights that has one
        element long the sample dimension that is 1.0. This element is the point
        where the 'surface' is.
    """
    cumulative_contribution = torch.cumsum(weights, dim=-1)
    depth_threshold = torch.tensor(depth_threshold, dtype=weights.dtype, device=device)
    opaqueness = cumulative_contribution >= depth_threshold
    false_padding = torch.zeros_like(opaqueness[..., :1],device = device)
    padded_opaqueness = torch.cat(
        [false_padding, opaqueness[..., :-1]], dim=-1)
    opaqueness_mask = torch.logical_xor(opaqueness, padded_opaqueness)
    opaqueness_mask = opaqueness_mask.type(weights.dtype)
    return opaqueness_mask

def compute_depth_index(device,weights, depth_threshold=0.5):
    """Compute the sample index of the median depth accumulation."""
    opaqueness_mask = compute_opaqueness_mask(device, weights, depth_threshold)
    return torch.argmax(opaqueness_mask, axis=-1)

def compute_depth_map(weights, z_vals, depth_threshold=0.5):
    """Compute the depth using the median accumulation.

    Note that this differs from the depth computation in NeRF-W's codebase!

    Args:
        weights: the density weights from NeRF.
        z_vals: the z coordinates of the samples.
        depth_threshold: the accumulation threshold which will be used as the depth
        termination point.

    Returns:
        A tensor containing the depth of each input pixel.
    """
    opaqueness_mask = compute_opaqueness_mask(weights, depth_threshold)
    return torch.sum(opaqueness_mask * z_vals, dim=-1)

if __name__ == '__main__':
    pass
    # todo unit test