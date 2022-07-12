import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser


from utils import load_ckpt #todo refactor
from hypernerf.model_utils import prepare_ray_dict, extract_rays_batch, concat_ray_batch
from hypernerf.models import NerfModel
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/root/autodl-tmp/ClipNeRF_base/nerf-pytorch/data/nerf_llff_data/room/',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='llff',
                        choices=['blender', 'llff'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[504,378],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=2048,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, 
                            default="/root/autodl-tmp/HyperNeRF-torch/ckpts/fangzhou_embed/epoch=8.ckpt",
                            help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    ###########################
    #### params for warp ####
    parser.add_argument('--use_warp', type=bool, default=True,
                        help='whether to use warping')
    parser.add_argument('--hyper_slice_out_dim', type=int, default=4,
                            help='The output dimension of the hypersheet mlp')
    parser.add_argument('--slice_method', type=str, default='bendy_sheet',
                            help='method to slice the hyperspace, must be used with warping',
                            choices=['bendy_sheet', 'none', 'axis_aligned_plane'])
    ###########################
    #### params for embedding ####
    parser.add_argument("--meta_GLO_dim",type=int,default=16,
                            help="the dimension used for GLO embedding of time")
    parser.add_argument("--share_GLO",type=bool,default=True,
                            help="When set true, all GLO embedding use the same model and key")
    parser.add_argument("--use_nerf_embedding",action="store_true",
                            help="whether to use the nerf embedding")
    parser.add_argument("--use_alpha_condition",action="store_true",
                            help="whether to use the alpha condition, must be used with use_nerf_embedding")
    parser.add_argument("--use_rgb_condition",action="store_true",
                            help="whether to use the rgb condition, must be used with use_nerf_embedding")                     

    parser.add_argument("--xyz_fourier",type=int,default=10,
                            help="the dimension used for fourier embedding of points xyz")
    parser.add_argument("--hyper_fourier",type=int,default=6,
                            help="the dimension used for fourier embedding of points hyper feature")
    parser.add_argument("--view_fourier",type=int,default=6,
                        help="the dimension used for fourier embedding of view dir ")

    return parser.parse_args()


@torch.no_grad()
def batched_inference(model,
                      rays_dict, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays_dict["origins"].shape[0]
    chunk = 1024
    results = defaultdict(list)
    extra_params = {
        'nerf_alpha': None,
        'warp_alpha': None,
        'hyper_alpha': None,
        'hyper_sheet_alpha': None,
    }
    for i in range(0, B, chunk):
        #for all rays in an image
        ray_dict_batch = extract_rays_batch(rays_dict, i, i+chunk)
        rendered_ray_chunks = model(ray_dict_batch, extra_params)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    # concatenate chunks
    for k, v in results.items():
        results[k] = concat_ray_batch(v)

    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh)}

    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)
    #!todo replace to real data
    NUM_IMG = 100 #!hardcoded, the total number of images in the dataset
    embeddings_dict = {'warp': [_ for _ in range(NUM_IMG)],
    'camera':[0],
    'appearance': [_ for _ in range(NUM_IMG)], 
    'time': [_ for _ in range(NUM_IMG)]}
    nerf = NerfModel(
                    embeddings_dict,
                    near = 0.0,
                    far=1.0, # todo assume ndc here
                    n_samples_coarse=args.N_samples,
                    n_samples_fine=args.N_importance,
                    hyper_slice_method = args.slice_method,
                    hyper_slice_out_dim = args.hyper_slice_out_dim,
                    use_warp = args.use_warp,
                    use_nerf_embed= args.use_nerf_embedding,
                    use_alpha_cond= args.use_alpha_condition,
                    use_rgb_cond= args.use_rgb_condition,
                    GLO_dim = args.meta_GLO_dim,
                    share_GLO = args.share_GLO,
                    xyz_fourier_dim = args.xyz_fourier,
                    hyper_fourier_dim = args.hyper_fourier,
                    view_fourier_dim= args.view_fourier,
                    
                    )
                    
    load_ckpt(nerf, args.ckpt_path, model_name='nerf')
    nerf.cuda().eval()

    imgs = []
    psnrs = []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        rays_dict = prepare_ray_dict(rays) # same times
        results = batched_inference(nerf, rays_dict,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back)

        img_pred = results['fine']['rgb'].view(h, w, 3).cpu().numpy()
        
        if args.save_depth:
            depth_pred = results['fine']['depth'].view(h, w).cpu().numpy()
            depth_pred = np.nan_to_num(depth_pred)
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            else:
                with open(f'depth_{i:03d}', 'wb') as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]
        
    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)
    
    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')