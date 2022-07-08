import os, sys

from opt import get_opts
import torch

from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict


# hypernerf
from hypernerf.models import NerfModel
from hypernerf.model_utils import append_batch, prepare_ray_dict, extract_rays_batch, concat_ray_batch

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
#debugging
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
# torch.autograd.set_detect_anomaly(True)

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)
        self.loss = loss_dict[hparams.loss_type]()

        #todo not used
        self.embeddings_dict = {'warp': [1,2,3], 'camera':[1,2,3], 'appearance': [1,2,3], 'time': [1,2,3]}
        self.nerf = NerfModel(
                            self.embeddings_dict,
                            near = 0.0,
                            far=1.0, # todo use ndc here
                            n_samples_coarse=hparams.N_samples,
                            n_samples_fine=hparams.N_importance,
                            noise_std=hparams.noise_std,
                            hyper_slice_method =  hparams.slice_method,
                            use_warp = hparams.use_warp,
                            )
                            #when use warp, remember to include the hyper sheet

        self.models = {'nerf': self.nerf}
        load_ckpt(self.nerf, hparams.weight_path, 'nerf')

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)


    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs
    

    def forward(self, rays_dict):
        """Do batched inference on rays using chunk."""
        B = rays_dict["origins"].shape[0]
        results = {"coarse": None, "fine": None}
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
            results = append_batch(results, self.nerf(ray_dict_batch, extra_params))
        # # concatenate chunks
        # for k, v in results.items():
        #     results[k] = concat_ray_batch(v)

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

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

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

            depth = visualize_depth(results[typ]["depth"].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)
            log['val_psnr'] = psnr(results[typ]["rgb"], rgbs)
        
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


if __name__ == '__main__':
    hparams = get_opts()

    system = NeRFSystem(hparams)
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=5)

    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    profiler = AdvancedProfiler()


    trainer = Trainer(
                      precision=hparams.precision,
                      amp_backend='native',
                      max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      enable_model_summary=True,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None,
                      strategy = 'ddp_sharded'if hparams.num_gpus>1 else None,
                    #   strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None,
                      val_check_interval=0.25,
                      )

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