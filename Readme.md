# HyperNeRF-pl


A Non-official pytorch lightning implementation of [HyperNeRF](https://arxiv.org/abs/2106.13228). The code structure follows the original implementation of [HyperNeRF](https://github.com/google/hypernerf), which is based on JAX. The training and dataloading code are taken from [NeRF-pl](https://github.com/kwea123/nerf_pl/tree/dev). Overall, this project does not aim at reproducing the original implementation, but rather to provide a working implementation of HyperNeRF. Pull requests & issues are welcome.

IMPORTANT： this is a working repo, the code is not finished yet.

## Installation
### Hardware
- OS: Ubuntu 18.04 LTS
- GPU: Tested with NVIDIA GTX 2080Ti with CUDA 11.5 (both single and multi-GPU)

### Software
1. Clone the repo: `git clone https://github.com/songrise/HyperNeRF-torch`

2. Create a virtual environment (we suggest Anaconda). `conda create -n hypernerf-torch python=3.7`. Then activate it by `conda activate hypernerf-torch`.

3. Install the dependencies. `pip install -r requirements.txt` 

## Training
Currently, we only support [LLFF-style](https://github.com/Fyusion/LLFF) dataset training. That is, we use the LLFF file structure for dataset loading, but the scene does not necessarily be static. If you already have a LLFF dataset, we do not require any changes and you can directly train the model. If you have a Nerfies dataset, you may need to convert it to LLFF format, follow the guide in [this repo](https://github.com/kwea123/nerf_pl/tree/dev). If you have a COLMAP dataset, convert it into [LLFF style](https://github.com/Fyusion/LLFF). If you have raw image sequence, run [COLMAP](https://github.com/colmap/colmap) for sparse reconstruction.

### [Optional] Data download
If you do not have your own llff style dataset, you can get one [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

### Training example
To train the model, you can use the following command:
```
python train.py \
   --dataset_name llff \
   --root_dir $LLFF_DIR \
   --N_importance 64 --img_wh $IMG_W #IMG_H \
   --num_epochs 30 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --use_warp=True --slice_method "bendy_sheet"\
   --exp_name exp
```

For multi-GPU training, simply add `--num_gpus 2` to the command.
For reference, it tooks around 7 hours to train 100k iteration and get a fairly good model (~30 psnr on our own dataset).
## Evaluation

To test the model, you can use the following command:
```
python eval.py --ckpt_path=$CKPT_PATH \
--scene_name="exp" --img_wh $IMG_W #IMG_H \
--root_dir "/exp"
```
This will generate a folder in `result` directory with the rendered images and a gif of the result. Note that you have to pass the same set of arguments as in the training command to load the model.

## Implementation details
This repo is a by-product of one of my research projects, therefore I do not have much time to precisely replicate the original implementation. Most notably, the following are different from the original implementation:

- SE3-field coded but not debugged yet. For warping, we currently use Translation field.
- We generate the metadata of the rays (e.g, warp embedding key) in run time, instead of reading it from the file.
- We use the original positional encoding scheme in NeRF, instead of the one used in HyperNeRF.
- We did not implement the code that loads Nerfies-style dataset

If you encounter any other problems, please feel free to contact me.

## Todo list
- [x] unit test hypernerf model
- [x] check the tensor shape with the jax imple
- [x] llff dataloading and traing code
- [x] add noise impl
- [x] add evaluation code.
- [x] switch to pytorch-lightning
- [x] test on llff dataset
- [ ] add nerfies dataset loading code
- [ ] experiment on the nerfies dataset
- [ ] test the SE3Field 
- [ ] Clean the code and add docs
