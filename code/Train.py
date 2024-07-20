
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandAffined,
    Rand3DElasticd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
    ToTensord,
    RandFlipd,
    RandRotate90d
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.nets import VNet
from monai.networks.nets import UNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.metrics import ConfusionMatrixMetric
from monai.metrics import MeanIoU
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import numpy as np
import tarfile
import shutil
import os
import glob
from tqdm import tqdm
from einops import rearrange,repeat,reduce
from utils import ramps, losses, utils

from Nets import *







import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='1-8-CBAM-ramp-pos2', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=40000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--patch_num', type=int, default=3, help='patch_num per sample')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
args = parser.parse_args(args=[])





iter_num = 0
lr_ = args.base_lr
train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"
save_path = "../saved_model/" + args.exp + "/"
if not os.path.exists(save_path):
        os.makedirs(save_path)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
base_lr = args.base_lr
labeled_bs = args.labeled_bs
patch_num = args.patch_num
B = args.batch_size*patch_num
labeled_bn = B//2


train_images = sorted(
    glob.glob(os.path.join('../raw/', "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join('../mask/', "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]  
train_files, val_files= data_dicts[:100], data_dicts[-42:]
set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd( 
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged( 
            keys=["image"],
            a_min=-210,
            a_max=290,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"), 
        RandCropByPosNegLabeld( 
            keys=["image", "label"],
            label_key="label",
            spatial_size=(112,112,80),
            pos=2,
            neg=1,
            num_samples=patch_num,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-210, a_max=290, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

set_determinism(seed=0)
labeled_idxs = list(range(20))
unlabeled_idxs = list(range(20, 100))
batch_sampler = utils.TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4
    )
train_loader = DataLoader(
    train_ds, 
    batch_sampler=batch_sampler, 
    num_workers=4, 
    pin_memory=True,
    #worker_init_fn=lambda worker_id: worker_init_fn(worker_id, args)
)
val_ds = CacheDataset(
    data=val_files, transform=val_transforms,cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)




def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)




def store_values(values, file_path):
    with open(file_path, 'a') as file:  
        file.write(f"{values[-1]}\n")



def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen





def create_model(net_type,ema=False):
        if net_type=='VNet':
            net = VNet(spatial_dims=3,in_channels=1,out_channels=2,act=('elu', {'inplace': True}),dropout_prob=0.5, dropout_dim=3,bias=False)
        if net_type == 'Dual_Vnet':
            net = VNet_dual_de(spatial_dims=3,in_channels=1,out_channels=2,act=('elu', {'inplace': True}),
                       dropout_prob_down=0.5, dropout_prob_up=(0.5, 0.5), dropout_dim=3,bias=False)
        if net_type == 'Dual_Att_Vnet':
             net = VNet_dual_atten_CBAM(spatial_dims=3,in_channels=1,out_channels=2,act=('elu', {'inplace': True}),
                       dropout_prob_down=0.5, dropout_prob_up=(0.5, 0.5), dropout_dim=3,bias=False)

        model = net.cuda()
        
        if ema:
            for param in model.parameters():
                param.detach_()
        
        return model


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_type='Dual_Att_Vnet'

model = create_model(net_type=net_type)
ema_model = create_model(net_type=net_type,ema=True)





from monai.losses import DiceCELoss
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from torch.optim import lr_scheduler
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
freeze_layers = [
    'de2_up_tr256', 'de2_up_tr128', 'de2_up_tr64', 'de2_up_tr32', 'de2_out_tr', 'de2_confi','cbam_block'
]
params_to_optimize_s1 = [
    param for name, param in model.named_parameters()
    if any(layer in name for layer in freeze_layers)
]
optimizer_s1 = torch.optim.SGD(params_to_optimize_s1, lr=0.01, momentum=0.9, weight_decay=0.0001)

params_to_optimize_s2 = [
    param for name, param in model.named_parameters()
    if not any(layer in name for layer in freeze_layers)
]
optimizer_s2 = torch.optim.SGD(params_to_optimize_s2, lr=0.01, momentum=0.9, weight_decay=0.0001)


dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)




mode='cosineAnn'
if mode=='cosineAnn':
    scheduler_s1=CosineAnnealingLR(optimizer_s1,T_max=8000*20,eta_min=0.001)
elif mode=='cosineAnnWarm':
    scheduler_s1=CosineAnnealingWarmRestarts(optimizer_s1,T_0=10,T_mult=2,eta_min=0.001)

mode='cosineAnn'
if mode=='cosineAnn':
    scheduler_s2=CosineAnnealingLR(optimizer_s2,T_max=8000*20,eta_min=0.001)
elif mode=='cosineAnnWarm':
    scheduler_s2=CosineAnnealingWarmRestarts(optimizer_s2,T_0=10,T_mult=2,eta_min=0.001)




if args.consistency_type == 'mse':
    consistency_criterion = losses.mse_loss
elif args.consistency_type == 'kl':
    consistency_criterion = losses.kl_loss
else:
    assert False, args.consistency_type






max_epochs = 8000
warm_up_epoch = 400
val_interval = 25
confi_interval = 1000
confi_iter_train = 5
best_metric = -1
best_metric_epoch = -1
best_metric_model = -1
best_metric_epoch_model  = -1
epoch_loss_values_s1 = []
epoch_loss_values_s2 = []
metric_values = []
metric_values_model = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)]) 
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)]) 

confidence_wce_weight = None



import time
import torch.nn.functional as F



def ema_model_wrapper(inputs):
    return ema_model(inputs)[0] 

def model_wrapper(inputs):
    return model(inputs)[0] 


for epoch_num in tqdm(range(max_epochs), ncols=70):
    model.train()
    ema_model.train()
    step = 0
    epoch_loss_s1 = 0
    epoch_loss_seg_s1 = 0
    epoch_loss_confi_s1 = 0

    epoch_loss_s2 = 0
    epoch_loss_seg_s2 = 0
    epoch_loss_consistency_s2 = 0
    epoch_consistency_dist_s2 = 0
    
    epoch_acc_0 = 0
    epoch_acc_1 = 0
    for i_batch, sampled_batch in enumerate(train_loader):
        time2 = time.time()
        step += 1
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        noise_1 = torch.clamp(torch.randn_like(volume_batch) * 0.2, -0.4, 0.4)
        inputs_1 = volume_batch + noise_1
        if epoch_num < warm_up_epoch:
            current_label = label_batch[:labeled_bn]
            prediction, confi = model(volume_batch[:labeled_bn])
            loss_seg = loss_function(prediction, current_label)
            predicted_classes = torch.argmax(prediction, dim=1)
            acc_mask = (predicted_classes == current_label.squeeze(1)).float()
            ecn_criterion = nn.CrossEntropyLoss(weight=torch.tensor(utils.weighted_binary_cross_entropy(acc_mask)).cuda())
            confi_loss = ecn_criterion(confi, acc_mask.long())
            epoch_loss_seg_s1 += loss_seg.item()
            epoch_loss_confi_s1 += confi_loss.item()
            loss = confi_loss
            optimizer_s1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_s1.step()
            scheduler_s1.step() 
            loss = loss_seg 
            optimizer_s2.zero_grad()
            loss.backward()
            optimizer_s2.step()
            scheduler_s2.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
        elif (epoch_num + 1) % confi_interval <confi_iter_train: 
            prediction, confi = model(inputs_1)
            predicted_classes = torch.argmax(prediction, dim=1)
            acc_mask = (predicted_classes == label_batch.squeeze(1)).float()
            ecn_criterion = nn.CrossEntropyLoss(weight=torch.tensor(utils.weighted_binary_cross_entropy(acc_mask[:labeled_bn])).cuda())
            confi_loss = ecn_criterion(confi[:labeled_bn], acc_mask[:labeled_bn].long())
            acc_0, acc_1 = utils.class_wise_accuracy(torch.argmax(confi[labeled_bn:], dim=1), acc_mask[labeled_bn:])
            epoch_acc_0 += acc_0
            epoch_acc_1 += acc_1
            loss = confi_loss
            epoch_loss_s1 += loss.item()
            epoch_loss_seg_s1 += loss_seg.item()
            epoch_loss_confi_s1 += confi_loss.item()
            optimizer_s1.zero_grad()
            loss.backward()
            optimizer_s1.step()
            scheduler_s1.step() 
        else: 
            prediction = torch.zeros([inputs_1.size(0), 2,112,112,80]).cuda()
            prediction[:labeled_bn], _ = model(inputs_1[:labeled_bn])
            prediction[labeled_bn:], _ = model(inputs_1[labeled_bn:])
            prob = F.softmax(prediction, dim=1)
            loss_seg = loss_function(prediction[:labeled_bn], label_batch[:labeled_bn])
            noise_2 = torch.clamp(torch.randn_like(volume_batch) * 0.05, -0.1, 0.1)
            inputs_2 = volume_batch + noise_2
            ema_pred, ema_confi = ema_model(inputs_2)
            ema_prob = F.softmax(ema_pred, dim=1)

            consistency_dist = consistency_criterion(prob, sharpening(ema_prob))
            confi_prob = F.softmax(ema_confi, dim=1)[:, 1:2]
            confi_weight = confi_prob.round()
            weighted_consistency_loss = (consistency_dist * confi_weight).mean()
            
            loss = loss_seg + weighted_consistency_loss*ramps.sigmoid_rampup(epoch_num, max_epochs)
            epoch_loss_s2 += loss.item()
            epoch_loss_seg_s2 += loss_seg.item()
            epoch_loss_consistency_s2 += weighted_consistency_loss.item()
            epoch_consistency_dist_s2 += consistency_dist.mean().item()
            optimizer_s2.zero_grad()
            loss.backward()
            optimizer_s2.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            scheduler_s2.step()
            iter_num = iter_num + 1
        
        

    if epoch_num >= warm_up_epoch and (epoch_num + 1) % val_interval == 0:
        ema_model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (112,112,80)
                sw_batch_size = 4
                val_outputs = sliding_window_inference( 
                    val_inputs, roi_size, sw_batch_size, ema_model_wrapper)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric) 
            store_values(metric_values, save_path + 'metric_value.csv')
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch_num + 1
                model_save_path = os.path.join(
                    save_path, 
                    f"epoch{best_metric_epoch}-{metric:.4f}.pth"
                )
                torch.save(ema_model.state_dict(), model_save_path)
                print("saved new best metric model")
            print(f"Epoch {epoch_num + 1} - Current metric: {metric}, Best metric: {best_metric} at Epoch {best_metric_epoch}")
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (112, 112, 80)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model_wrapper)  
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values_model.append(metric)  
            store_values(metric_values_model, save_path + 'metric_value_model.csv')  

            if metric > best_metric_model:  
                best_metric_model = metric
                best_metric_epoch_model = epoch_num + 1
                model_save_path = os.path.join(
                    save_path, 
                    f"model_epoch{best_metric_epoch_model}-{metric:.4f}.pth"  
                )
                torch.save(model.state_dict(), model_save_path)
                print("saved new best metric model for standard model")

            print(f"Epoch {epoch_num + 1} - Current metric for standard model: {metric}, Best metric: {best_metric_model} at Epoch {best_metric_epoch_model}")



