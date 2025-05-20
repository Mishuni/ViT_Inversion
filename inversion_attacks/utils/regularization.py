import torch
from einops import rearrange


def total_variance(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def label_matching(pred, label):
    
    onehot_label = torch.eye(pred.shape[-1])[label.long().cpu()]
    onehot_label = onehot_label.to(pred.device)
    return torch.sqrt(torch.sum((pred - onehot_label) ** 2))


def group_consistency(x, group_x):
    mean_group_x = sum(group_x) / len(group_x)
    return torch.norm(x - mean_group_x, p=2)


def bn_regularizer(feature_maps, bn_layers):
    bn_reg = 0
    for i, layer in enumerate(bn_layers):
        fm = feature_maps[i]
        if len(fm.shape) == 3:
            dim = [0, 2]
        elif len(fm.shape) == 4:
            dim = [0, 2, 3]
        elif len(fm.shape) == 5:
            dim = [0, 2, 3, 4]
        bn_reg += torch.norm(fm.mean(dim=dim) - layer.state_dict()["running_mean"], p=2)
        bn_reg += torch.norm(fm.var(dim=dim) - layer.state_dict()["running_var"], p=2)
    return bn_reg


def patch_regularizer(x,patch_size=16):
    ## TODO: patch prior loss
    temp = x.detach() #.permute(0, 2, 3, 1)
    patches = rearrange(temp, 'b c (h s1) (w s2) -> b h w s1 s2 c',  s1 = patch_size, s2 = patch_size)
    height_patch_num = patches.shape[1]
    width_patch_num = patches.shape[2]
    pp_reg = 0
    for i in range(1,height_patch_num):
        pp_reg += torch.norm(patches[:,i,:,:,:] - patches[:,i-1,:,:,:], p=2)
    for i in range(1,width_patch_num):
        pp_reg += torch.norm(patches[:,:,i,:,:] - patches[:,:,i-1,:,:], p=2)
    # print(pp_reg)
    return pp_reg