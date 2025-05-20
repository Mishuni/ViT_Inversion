import argparse, csv, sys, os
sys.path.append('../')
import numpy as np
from matplotlib import pyplot as plt
from IQA_pytorch import SSIM 
import lpips 

import torch
import torch.nn as nn
import torchvision.models as models
from vit_pytorch import ViT

from inversion_attacks import GradientInversion_Attack
from inversion_attacks.utils.utils import NumpyDataset, ConservativeStrategy, plot
from inversion_attacks.utils.models import LeNet, ConvNet, MoCo
from inversion_attacks.utils.evaluation_metrics import psnr
from inversion_attacks.utils.consts import *

def evaluation_matric(net,loss,output,reconstructed_label,ground_truth,classes,save_file_path,mean_std,device='cpu',index=0):
    output.requires_grad=False
    ground_truth.requires_grad=False
    
    ds,dm = mean_std
    test_mse = (output.detach() - ground_truth).pow(2).mean()
    feat_mse = (net(output.detach())- net(ground_truth)).pow(2).mean()  
    test_psnr = psnr(output, ground_truth, factor=1/ds)

    D = SSIM(channels=3)
    test_ssim = D(output.detach(), ground_truth, as_loss=False).mean()

    lpips_loss = lpips.LPIPS(net='vgg', spatial=True).to(device)
    lpips_score = lpips_loss.forward(ground_truth, output.detach()).mean()

    result_list  = [index,loss.item(), test_mse.item(), feat_mse.item(), test_psnr, test_ssim.item(), lpips_score.item(),str(classes[reconstructed_label][0])]

    plt.figure(figsize=(7,8))
    plot(output,ds,dm)
    plt.title(f"loss: {loss:2.4f} | MSE: {test_mse:2.4f} \n"
                    f"PSNR: {test_psnr:3.2f} | FMSE: {feat_mse:2.4e} ")
    plt.savefig(os.path.join(save_file_path, f'result_{str(index)}.png')) 

    return result_list

def extract_bn(from_model):
    bn_reg_layers = []
    for module in from_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_reg_layers.append(module)
    print("# of the Batch norm layers: ",len(bn_reg_layers))
    return bn_reg_layers

def get_image_prior_model(checkpoint_path,prior_model_arch,device):
    checkpoint = torch.load(checkpoint_path)
    image_prior_model = MoCo(models.__dict__[prior_model_arch],128,65536,0.999,0.2,True) 
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            k_no_prefix = k[len("module."):]
            state_dict[k_no_prefix] = state_dict[k]
            state_dict[k_no_prefix.replace('encoder_q', 'encoder_k')] = state_dict[k]
        del state_dict[k]

    image_prior_model.load_state_dict(state_dict, strict=False)
    image_prior_model.to(device)
    return image_prior_model

def extracted_wanted_labels(label_shape,label_results):
        if len(label_shape) > 1: return [torch.argmax(res).item() for res in label_results]
        else : return label_results