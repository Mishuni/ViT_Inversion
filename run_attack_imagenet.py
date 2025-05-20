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
from inversion_attacks.utils.utils import NumpyDataset, ConservativeStrategy, plot, multi_plot
from inversion_attacks.utils.models import LeNet, ConvNet, MoCo
from inversion_attacks.utils.evaluation_metrics import psnr
from inversion_attacks.utils.consts import *
from inversion_attacks.utils.dataloader import construct_dataloaders
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default="resnet18", help="model architecture")
parser.add_argument('--attack-name','-a', type=str, default="gs", help="attack mode 'dlg', 'idlg', 'gs', 'cpl', 'gi', 'gv'")
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--max-iters','-i', type=int, default=10000)
parser.add_argument('--batch-size','-b', type=int, default=1)
parser.add_argument('--gpu','-g', type=int, default=5)
parser.add_argument('--with-labels', action='store_true')

args = parser.parse_args()

if __name__=="__main__":
    torch.manual_seed(5)
    num_classes = 1000
    channel = 3
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else "cpu"
    setup = dict(device=device, dtype=torch.float)  
    defs = ConservativeStrategy()
    batch_size = args.batch_size
    data_path = '/data2/imagenet2012'
    attack_name = args.attack_name # 'dlg', 'idlg', 'gs', 'cpl', 'gi', 'gv'
    model_arch = args.arch.lower() # lenet, resnet18, resnet50, vit, convnet
    max_iterations = args.max_iters
    attack_lr = args.lr
    dm = torch.as_tensor([0.5, 0.5, 0.5], **setup)[:, None, None]
    ds = torch.as_tensor([0.5, 0.5, 0.5], **setup)[:, None, None] 
    with_labels = args.with_labels

    save_data_path = f"./results/{attack_name}/{model_arch.lower()}/{batch_size}_{max_iterations}_{attack_lr}_{with_labels}"
    save_file_path = os.path.join(save_data_path)
    if os.path.exists(save_file_path) is not True:
        os.makedirs(save_file_path)
    print(f"The result files will saved at {save_file_path}")

    loss_fn, dataloader, validloader =  construct_dataloaders('ImageNet', defs,data_path=data_path)
    classes=dataloader.dataset.classes
    
    ## load ground_truth images
    if batch_size==1:
        img_idx = 8112
        ground_truth, labels = validloader.dataset[img_idx]
        labels = torch.as_tensor((labels,), device=device)
        ground_truth = ground_truth.to(**setup).unsqueeze(0)
    else:
        for i, (ground_truth,labels) in enumerate(dataloader):
            ground_truth = ground_truth.to(device)
            labels = labels.to(device)
            multi_plot(ground_truth,ds,dm,labels,dataloader.dataset.classes)
            plt.savefig(os.path.join(save_file_path, 'ground_truth.png')) 
            break
    shape_img=tuple(ground_truth[0].shape)

    ## model load
    if model_arch=='resnet50' or  model_arch=='resnet18':
        net = getattr(models, model_arch)()
    elif model_arch=='lenet':
        net = LeNet(channel=channel, hideen=37632, num_classes=num_classes)
    elif model_arch=='convnet':
        net = ConvNet(num_classes=num_classes)
    elif model_arch=='vit':
        net = ViT(
            image_size = 224,
            patch_size = 16,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    net = net.to(device)

    ## init fake x
    x_init = torch.randn((batch_size,) + (shape_img), requires_grad=True, device=device)

    ## extract gradient
    net.eval()
    net.zero_grad()
    ground_truth.requires_grad = True

    pred = net(ground_truth)
    loss, _, _ = loss_fn(pred, labels)
    received_gradients = torch.autograd.grad(loss, net.parameters())
    received_gradients = [cg.detach() for cg in received_gradients]
    print(f"# of Received Gradients : {len(received_gradients)}")

    ## define attacker
    if attack_name == 'dlg':
        attacker = GradientInversion_Attack(
                net, 
                shape_img, 
                mean_std=(dm,ds), 
                lr=attack_lr,#1.0, 
                log_interval=100, 
                num_iteration=max_iterations,#3000, 
                distancename="l2",
                device=device,early_stopping=5000,
                lr_decay=True,
                optimize_label=True,
                save_file_path=save_file_path
            )
    elif attack_name == 'gs':
        attacker = GradientInversion_Attack(
                    net,
                    shape_img,
                    mean_std=(dm,ds),
                    lr=attack_lr, #lr=1.0,
                    log_interval=100,
                    num_iteration=max_iterations,#10000,
                    tv_reg_coef=1e-6,
                    distancename="cossim",
                    optimizer_class=torch.optim.Adam,
                    device=device,early_stopping=5000,
                    optimize_label=False,
                    lr_decay=True,
                    save_file_path=save_file_path
                    # clamp_range=(0,1)
                )
    elif attack_name == 'idlg':
        attacker = GradientInversion_Attack(
                net,
                shape_img, 
                mean_std=(dm,ds),
                lr=attack_lr,#0.001,
                log_interval=100,
                num_iteration=max_iterations,#3000,
                distancename="l2",
                optimize_label=False,
                optimizer_class=torch.optim.SGD,
                lr_decay= True,
                device=device,early_stopping=5000,
                save_file_path=save_file_path
            )
    elif attack_name == 'cpl':
        attacker = GradientInversion_Attack(
                net,
                shape_img,
                mean_std=(dm,ds),
                lr=attack_lr,#1.0,
                log_interval=100,
                num_iteration=max_iterations,#500,
                distancename="l2",
                optimize_label=False,
                lm_reg_coef=0.01,
                device=device,early_stopping=5000,
                save_file_path=save_file_path
            )
    elif attack_name == 'gi':
        group_num = 2
        bn_reg_layers = extract_bn(net)
        attacker = GradientInversion_Attack(
            net,
            shape_img,
            mean_std=(dm,ds),
            num_iteration=max_iterations,#20000,
            lr=attack_lr,#0.1,
            optimizer_class=torch.optim.Adam,
            log_interval=100,
            distancename="l2",
            optimize_label=True,
            bn_reg_layers=bn_reg_layers,
            group_num=group_num,
            tv_reg_coef=0.0001,
            l2_reg_coef=0.000001,
            bn_reg_coef=0.1,
            gc_reg_coef=0.001,
            lr_decay= True,
            device=device,early_stopping=1000,
            save_file_path=save_file_path
        )
    elif attack_name == 'gv':
        # GradViT
        group_num = 2
        image_prior_model=get_image_prior_model('./demodata/moco_v2_200ep_pretrain.pth.tar','resnet50',device)
        image_prior_model.eval()
        image_prior_model.zero_grad()
        ip_bn_reg_layers = extract_bn(image_prior_model)
        bn_reg_layers = extract_bn(net)

        attacker = GradientInversion_Attack(
            net,
            shape_img,
            mean_std=(dm,ds),
            num_iteration= max_iterations, #10000,#20000,
            lr=attack_lr, #0.1,
            optimizer_class=torch.optim.Adam,
            log_interval=100,
            distancename="l2",
            optimize_label=False,#True,
            bn_reg_layers=bn_reg_layers,
            group_num=group_num,
            gc_reg_coef=0.01,
            lr_decay= True,
            device=device,early_stopping=5000,
            save_file_path=save_file_path,
            image_prior_model=image_prior_model,
            ip_bn_reg_layers=ip_bn_reg_layers,
            ip_reg_coef=0.1,
            pp_reg_coef=0.0001,
            patch_size=16,
            ep_reg_coef=0.0001,
            loss_scheduler=True
        )

    ## start attack
    num_seeds=1
    attacker.reset_seed(num_seeds)
    if attack_name == 'gi' or attack_name == 'gv': 
        result = attacker.group_attack(received_gradients, batch_size=batch_size)
    else :
        if with_labels:
            result = attacker.attack(received_gradients,init_x=x_init,labels=labels,batch_size=batch_size)
        else:
            result = attacker.attack(received_gradients,init_x=x_init,batch_size=batch_size)

    ## evaluation and save result figures
    losses = attacker.get_best_loss()
    column_list = ['trial','loss','mse', 'feat_mse', 'psnr', 'ssim', 'lpips','label']
    results_list =[]
    if batch_size==1:
        if attack_name =="gi" or attack_name =="gv":
            result_label = extracted_wanted_labels(result[1][0].shape,result[1])
            last_results = result[2]
            for i,participant in enumerate(result[0]):
                results_list.append(evaluation_matric(net,losses[i],participant,result_label[i],ground_truth,classes,save_file_path,(dm,ds),device,index=i))
                results_list.append(evaluation_matric(net,losses[i],last_results[0][i],result_label[i],ground_truth,classes,save_file_path,(dm,ds),device,index=f'{i}_last'))
        else:
            result_label = extracted_wanted_labels(result[1].shape,result[1])
            results_list.append(evaluation_matric(net,losses,result[0],result_label[0],ground_truth,classes,save_file_path,(dm,ds),device))
    else:
        origin_labels = labels.tolist()
        #TODO : multi-batch evaluation
        if attack_name =="gi" or attack_name =="gv":
            result_labels = []
            for worker_id in range(group_num):
                result_labels.append(extracted_wanted_labels(result[1][worker_id].shape,result[1][worker_id]))
            for worker_id in range(group_num):
                result[0][worker_id].requires_grad=False
                multi_plot(result[0][worker_id],ds,dm,labels=result_labels[worker_id],classes=dataloader.dataset.classes)
                plt.savefig(os.path.join(save_file_path, f'result_{worker_id}.png')) 
                for i,current_label in enumerate(result_labels[worker_id]):
                    try:
                        found = origin_labels.index(current_label)
                    except:
                        continue
                    output = result[0][worker_id][i].detach().unsqueeze(0)
                    last_output = result[2][0][worker_id][i].detach().unsqueeze(0)
                    matched_ground_truth = ground_truth[found].detach().unsqueeze(0)
                    results_list.append(evaluation_matric(net,losses[worker_id],output,current_label,matched_ground_truth,classes,save_file_path,(dm,ds),device,index=f'worker{worker_id}_{i}'))
                    results_list.append(evaluation_matric(net,losses[worker_id],last_output,current_label,matched_ground_truth,classes,save_file_path,(dm,ds),device,index=f'worker{worker_id}_{i}_last'))
        else:
            result_label = extracted_wanted_labels(result[1].shape,result[1])
            for i,current_label in enumerate(result_label):
                try:    found = origin_labels.index(current_label)
                except: continue
                output = result[0][i].detach().unsqueeze(0)
                matched_ground_truth = ground_truth[found].detach().unsqueeze(0)
                results_list.append(evaluation_matric(net,losses,output,current_label,matched_ground_truth,classes,save_file_path,(dm,ds),device,index=f'batch_{i}'))
            multi_plot(result[0],ds,dm,labels=result[1],classes=classes)
            plt.savefig(os.path.join(save_file_path, 'result.png')) 
        
    ## write result into csv file
    with open(os.path.join(save_file_path, f'results.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(column_list)
            for i in range(len(results_list)):
                wr.writerow(results_list[i])
            avg = list(np.mean([r[1:-1] for r in results_list],axis=0))
            avg.insert(0,'Averaged')
            wr.writerow(avg)
