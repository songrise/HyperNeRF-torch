import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F

def sample_along_rays(origins, directions, num_coarse_samples, near, far,
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

    t_vals = torch.linspace(0., 1., num_coarse_samples,device=origins.device)
    if not use_linear_disparity:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    if use_stratified_sampling:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]],dim= -1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand([batch_size, num_coarse_samples],device=origins.device)
        z_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast z_vals to make the returned shape consistent.
        z_vals = z_vals[None, ...].expand(batch_size, num_coarse_samples)

    return (z_vals, (origins[..., None, :] +
                    z_vals[..., :, None] * directions[..., None, :]))

def volumetric_rendering(rgb,
                         sigma,
                         z_vals,
                         dirs,
                         use_white_background,
                         sample_at_infinity=True,
                         eps=1e-5):
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

    last_sample_z = 1e7 if sample_at_infinity else 1e-7 # in fp16, min_value is around 1e-8
    last_sample_z = torch.tensor(last_sample_z, device=rgb.device)

    #aka delta
    dists = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1],
        last_sample_z.expand(z_vals[..., :1].shape)
    ], dim = -1)

    dists = dists * torch.norm(dirs.unsqueeze(1), dim=-1)
    alpha = 1.0 - torch.exp(-sigma * dists)
    # Prepend a 1.0 to make this an 'exclusive' cumprod as in `tf.math.cumprod`.
    accum_prod = torch.cat([
        torch.ones_like(alpha[..., :1],device=rgb.device),
        torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
    ], dim=-1)
    
    weights = alpha * accum_prod

    rgb = torch.sum(weights[..., None] * rgb,dim = -2)

    exp_depth = torch.sum(weights * z_vals,dim=-1)
    med_depth = compute_depth_map(weights, z_vals)
    acc = torch.sum(weights,dim=-1)
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

# def piecewise_constant_pdf(bins, weights, num_coarse_samples,
#                            use_stratified_sampling):
#     """Piecewise-Constant PDF sampling.

#     Args:

#         bins: jnp.ndarray(float32), [batch_size, n_bins + 1].
#         weights: jnp.ndarray(float32), [batch_size, n_bins].
#         num_coarse_samples: int, the number of samples.
#         use_stratified_sampling: bool, use use_stratified_sampling samples.

#     Returns:
#         z_samples: jnp.ndarray(float32), [batch_size, num_coarse_samples].
#     """
#     eps = 1e-5

#     # Get pdf
#     weights = weights + eps  # prevent nans
#     pdf = weights / weights.sum(dim=-1, keepdims=True)
#     cdf = torch.cumsum(pdf, dim=-1)
#     cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1], device = weights.device), cdf], dim=-1)

#     # Take uniform samples
#     if use_stratified_sampling:
#         u = torch.rand(list(cdf.shape[:-1]) + [num_coarse_samples],device = weights.device)
#     else:
#         u = torch.linspace(0., 1., num_coarse_samples, device = weights.device)
#         new_shape =  cdf.shape[:-1] + [num_coarse_samples]
#         u = u.expand(*new_shape)
#     # Invert CDF. This takes advantage of the fact that `bins` is sorted.
#     mask = (u[..., None, :] >= cdf[..., :, None])

#     def minmax(x):
#         #todo check whether keep dim
#         x0,_ = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), dim=-2)
#         x1,_ = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), dim=-2)
#         x0 = torch.minimum(x0, x[..., -2:-1])
#         x1 = torch.maximum(x1, x[..., 1:2])
#         return x0, x1

#     bins_g0, bins_g1 = minmax(bins)
#     cdf_g0, cdf_g1 = minmax(cdf)

#     denom = (cdf_g1 - cdf_g0)
#     one_ = torch.scalar_tensor(1., device = weights.device)
#     denom = torch.where(denom < eps, one_, denom)
#     t = (u - cdf_g0) / denom
#     z_samples = bins_g0 + t * (bins_g1 - bins_g0)

#     return z_samples

def piecewise_constant_pdf(bins, weights, num_coarse_samples,
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
    N_rays, N_samples_ = weights.shape
    N_importance = num_coarse_samples
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if not use_stratified_sampling:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])

    return samples.detach()
    
def sample_pdf(bins, weights, origins, directions, z_vals,
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
    z_samples = piecewise_constant_pdf(bins, weights, num_coarse_samples,
                                        use_stratified_sampling)
    # Compute united z_vals and sample points
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
    #! shape [N,3] []
    return z_vals, (
        origins[..., None, :] + z_vals[..., None] * directions[..., None, :])
    
def posenc_orig(x, N_freqs,log_scale = True):
    """the encoding scheme used in the original NeRF paper"""
    batch_shape = x.shape[:-1]
    if log_scale:
        freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs,device = x.device)
    else:
        freq_bands = torch.linspace(0, N_freqs-1, N_freqs,device = x.device)
    funcs = [torch.sin, torch.cos]
    out = [x]
    for freq in freq_bands:
            for func in funcs:
                out = out + [func(freq*x)]
    return torch.cat(out, -1)

def get_posenc_ch_orig(in_ch, N_freq,log_scale = True):
    """get the channels of the posenc."""
    temp = torch.ones(1,1,in_ch).cuda()
    enc = posenc_orig(temp, N_freq)
    return enc.shape[-1]


def posenc(x, min_deg, max_deg, use_identity=False, alpha=None):
    """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1]."""
    batch_shape = x.shape[:-1]
    scales = 2.**torch.linspace(min_deg, max_deg, steps=max_deg-min_deg,device = x.device)
    # (*, F, C).
    xb = x[..., None, :] * scales[:, None]
    # (*, F, 2, C).
    four_feat = torch.sin(torch.stack((xb, xb + 0.5*3.1415926),dim = -2))
    #TODO temp disabled for debugging
    # if alpha is not None:
    #     window = posenc_window(x.device, min_deg, max_deg, alpha)
    #     four_feat = window[..., None, None] * four_feat

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

def posenc_window(device, min_deg, max_deg, alpha):
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
    bands = torch.linspace(min_deg, max_deg, steps=max_deg-min_deg,device = device)
    x = torch.clamp(alpha - bands, 0.0, 1.0)
    return 0.5 * (1 + torch.cos(3.1416926 * x + 3.1416926))

def noise_regularize(raw, noise_std, use_stratified_sampling):
    """Regularize the density prediction by adding gaussian noise.

    Args:

        raw: Dict[torch.Tensor], contains the rgb and density predictions.
        noise_std: float, std dev of noise added to regularize sigma output.
        use_stratified_sampling: add noise only if use_stratified_sampling is True.

    Returns:
        raw: jnp.ndarray(float32), [batch_size, num_coarse_samples, 4], updated raw.
    """
    if (noise_std is not None) and noise_std > 0.0 and use_stratified_sampling:
        # noise = torch.zeros_like(raw['alpha']).normal_(0, noise_std)
        noise = torch.randn(raw['alpha'].shape,
                device = raw['alpha'].device,dtype=raw['alpha'].dtype) * noise_std
        raw["alpha"] = raw["alpha"] + noise
    return raw

def compute_opaqueness_mask(weights, depth_threshold=0.5):
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
    depth_threshold = torch.tensor(depth_threshold, dtype=weights.dtype,device = weights.device )
    opaqueness = cumulative_contribution >= depth_threshold
    false_padding = torch.zeros_like(opaqueness[..., :1],device = weights.device)
    padded_opaqueness = torch.cat(
        [false_padding, opaqueness[..., :-1]], dim=-1)
    opaqueness_mask = torch.logical_xor(opaqueness, padded_opaqueness)
    opaqueness_mask = opaqueness_mask.type(weights.dtype)
    return opaqueness_mask

def compute_depth_index(weights, depth_threshold=0.5):
    """Compute the sample index of the median depth accumulation."""
    opaqueness_mask = compute_opaqueness_mask(weights, depth_threshold)
    return torch.argmax(opaqueness_mask, dim=-1)

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


def prepare_ray_dict(rays:torch.Tensor)->dict:
    """convert the nerf-pl ray tensor into a dictionary, so that it is 
       compatible with the hypernerf forward pass.
    
        Args:
        rays: the ray tensor, with shape (batch_size*num_samples, 8),
                 8 for raydir,orig,near,far.
        Returns:
        rays_dict: a dictionary containing the ray information. Contains:
                    'origins': the ray origins.
                    'directions': unit vectors which are the ray directions.
                    'viewdirs': (optional) unit vectors which are viewing directions.
                    'metadata': a dictionary of metadata indices e.g., for warping.
    """
    #TODO currently asuming the rays are of the same near, far.
    #if the last dim is 9, then the indices are included
    use_meta = rays.shape[-1] == 9

    if len(rays.shape) > 2:
        #if in [B, N, 8] format, flatten
        rays = rays.view(-1, 8)
    B = rays.shape[0]
    orig = rays[:,:3]
    dir = rays[:,3:6]
    near = rays[0,6]
    far = rays[0,7]
    idx = torch.ones((B,1),dtype=torch.long,device=rays.device) #dummy index
    if use_meta:
        idx = rays[:,8].type(torch.long)
    #todo: temporarily forge the metadata

    metadata= {'warp': idx.clone(),
                        'camera':  idx.clone(),
                        'appearance':  idx.clone(),
                        'time': idx.clone()}
    
    return {"origins": orig,
            "directions": dir,
            "viewdirs": None,
            "metadata": metadata}
   

def extract_rays_batch(rays:dict,start:int,end:int,drop_last=True)->dict:
    """
        extract ray batches from the ray dict.
        Args:
        rays: the ray dict.
        start: the start index of the batch.
        end: the end index of the batch.
        drop_last: if true, the last batch will be dropped.
        Returns:
        rays_batch: the extracted batch.
    """
    rays_batch = {k:None for k in rays.keys()}
    for key in rays.keys():
        if key == 'metadata':
            metadata_ = {k:None for k in rays[key].keys()}
            for k,v in rays[key].items():
                if v is not None:
                    metadata_[k] = v[start:end]
            rays_batch[key] = metadata_
        else:
            if rays[key] is not None:
                rays_batch[key] = rays[key][start:end]
            
    return rays_batch

def append_batch(all_ret,batch)->dict:
    """append a result batch to all result dict"""
    for k,v in all_ret.items():
        if v is None:
            all_ret[k] = batch[k]
        else:
            #append to dict
            for kk,vv in batch[k].items():
                if vv is not None:
                    all_ret[k][kk] = torch.cat([all_ret[k][kk],vv],dim=0)
    return all_ret

def concat_ray_batch(rays: list) -> dict:
    """
    concatenate a list of dictionary

    Args:
        rays: a list of ray dictionary
    
    Returns:
        a dictionary containing the concatenated rays tensor.
    """
    result = {k:None for k in rays[0].keys()}
    for c in rays:
        for k,v in c.items():
            if result[k] is None:
                result[k] = v
            else:
                result[k] = torch.cat([result[k],v],dim=0)
    return result





if __name__ == '__main__':
    pass
    # todo unit test