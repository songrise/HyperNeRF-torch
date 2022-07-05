import os, sys

from opt import get_opts
import torch

from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
# from models.nerf import Embedding, NeRF
# from models.rendering import render_rays

# hypernerf
from hypernerf.models import NerfModel
from hypernerf.model_utils import prepare_ray_dict, extract_rays_batch, concat_ray_batch

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger

#debugging
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
torch.autograd.set_detect_anomaly(True)

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()

        #todo not used
        self.embeddings_dict = {'warp': [1,2,3], 'camera':[1,2,3], 'appearance': [1,2,3], 'time': [1,2,3]}
        self.nerf = NerfModel(
                            self.embeddings_dict,
                            near = 0.0,
                            far=1.0, # todo use ndc here
                            n_samples_coarse=self.hparams.N_samples,
                            n_samples_fine=self.hparams.N_importance,
                            noise_std=self.hparams.noise_std,
                            hyper_slice_method = 'bendy_sheet'
                            )
                            #when use warp, remember to include the hyper sheet

        self.models = [self.nerf]

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs
    

    def forward(self, rays_dict):
        """Do batched inference on rays using chunk."""
        B = rays_dict["origins"].shape[0]
        results = defaultdict(list)
        #todo for debugging purposes
        # todo check the value
        extra_params = {
            'nerf_alpha': None,
            'warp_alpha': None,
            'hyper_alpha': None,
            'hyper_sheet_alpha': None,
        }
        for i in range(0, B, self.hparams.chunk):
            #for all rays in an image
            ray_dict_batch = extract_rays_batch(rays_dict, i, i+self.hparams.chunk)
            rendered_ray_chunks = self.nerf(ray_dict_batch, extra_params)
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]
        # concatenate chunks
        for k, v in results.items():
            results[k] = concat_ray_batch(v)

        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        rays_dict = prepare_ray_dict(rays)
        results = self(rays_dict)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[typ]['rgb'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        rays_dict = prepare_ray_dict(rays)
        results = self(rays_dict)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[typ]["rgb"].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            with open('depth.pkl', 'wb') as f:
                import pickle
                pickle.dump(results[typ]["depth"].view(H, W), f)

            depth = visualize_depth(results[typ]["depth"].view(H, W)) # (3, H, W)
            #dump depth for debugging                
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        log['val_psnr'] = psnr(results[typ]["rgb"], rgbs)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}',
                                                                '{epoch:d}'),
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=5,)

    logger = TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )
    profiler = AdvancedProfiler()
    trainer = Trainer(precision = 32,
                      max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler=True,
                      val_check_interval=0.2)
    # trainer = Trainer(
    #                   max_epochs=hparams.num_epochs,
    #                   checkpoint_callback=checkpoint_callback,
    #                   resume_from_checkpoint=hparams.ckpt_path,
    #                   logger=logger,
    #                   early_stop_callback=None,
    #                   weights_summary=None,
    #                   progress_bar_refresh_rate=1,
    #                   gpus=[0,1],
    #                   distributed_backend='ddp',
    #                   num_sanity_val_steps=0,
    #                   benchmark=True,
    #                   profiler=hparams.num_gpus==1)

    trainer.fit(system)