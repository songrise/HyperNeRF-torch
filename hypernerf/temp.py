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
print(a.shape, b.shape)