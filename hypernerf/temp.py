#%%
import torch
a = torch.rand(1,3,1,3)
b = a + 0.5
c = torch.stack([a,b],dim=1)
# %%
a = torch.rand(1,1,1)
b = torch.rand(1,1,3)
c = torch.cat([a,b],dim=-1)
# %%
v = torch.rand(1,1,4)
a =  v[..., :3]
b =  v[..., -1:]
noise = torch.randn(v.shape,device = v.device) * 1.0
#  %%
import pickle
import torch
import matplotlib.pyplot as plt
from PIL import Image

with open("../depth.pkl","rb") as f:
    depth = pickle.load(f)
    depth = depth.detach().cpu()
    depth = depth.squeeze()
    
    plt.imshow(depth)
    #calculate the mean depth
    mean_depth = torch.mean(depth)
# %%
