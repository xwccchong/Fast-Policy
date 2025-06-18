import imp
import cv2
import torch
import pdb
import random
import copy
import json
import inspect
import os
import logging
import warnings
import time
import datetime
import PIL.Image
import einops
import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from copy import deepcopy
from tqdm import tqdm
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, Tuple, List, Callable, Dict, Any
from scipy.ndimage import gaussian_filter1d
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.models.unet_2d_condition import UNet2DConditionOutput, logger
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    get_down_block,
    get_up_block,
)
from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from consistency_policy.utils import append_dims, reduce_dims
from .coefficient import noise_compensate_ddim_c

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def warpped_global_feature(sample, step):
    b, dim = sample.shape # (B,input_dim)
    processed_batches = []
    # sample_chunks = sample.chunk(b)
    # for batch in sample_chunks:
    #     batch = batch.repeat(step, 1)
    #     processed_batches.append(batch)
    for _ in range(step):
        copy_sample = sample.repeat(1,1)
        processed_batches.append(copy_sample)
    return torch.cat(processed_batches, dim=0)
    # return torch.cat(processed_batches, dim=1)

def warpped_feature(sample, step):
    """
    sample: [ batch_size x out_channels x horizon ]
    step: timestep span
    """
    # print(f"sample.shape:{sample.shape}")
    # b, dim, T = sample.shape # (B,T,input_dim)
    # uncond_fea, cond_fea = sample.chunk(2)
    # uncond_fea = uncond_fea.repeat(step,1,1,1) # (step * bs//2) * dim * h *w
    # cond_fea = cond_fea.repeat(step,1,1,1) # (step * bs//2) * dim * h *w
    # return torch.cat([uncond_fea, cond_fea])
    
    b, dim, T = sample.shape # (B,T,input_dim)
    processed_batches = []
    # sample_chunks = sample.chunk(b)
    # for batch in sample_chunks:
    #     batch = batch.repeat(step, 1, 1)
    #     processed_batches.append(batch)
    for _ in range(step):
        copy_sample = sample.repeat(1,1,1)
        processed_batches.append(copy_sample)
    return torch.cat(processed_batches, dim=0)
    # return torch.cat(processed_batches, dim=1)

def warpped_skip_feature(block_samples, step):
    down_block_res_samples = []
    for sample in block_samples:
        sample_expand = warpped_feature(sample, step)
        down_block_res_samples.append(sample_expand)
    return tuple(down_block_res_samples)

def warpped_timestep(timesteps, bs):
    """
    timestpes: list, such as [981, 961, 941]  bs= sample.shape[0] [95,96]
    """
    # semi_bs = bs//2
    semi_bs = bs//2
    ts = []
    # print(timesteps)
    for timestep in timesteps:
        timestep = timestep[None]
        texp = timestep.expand(semi_bs)
        ts.append(texp)
    # print(ts)
    timesteps = torch.cat(ts) # 拼接一组张量
    # print(timesteps)
    return timesteps.repeat(2,1).reshape(-1) # 复制后自动形成新的形状

def Fourier_filter(x, low_scale, high_scale, flag=False):
    if flag == True:
        dtype = x.dtype # torch.float32
        device = x.device
        x = x.type(torch.float32)
        # print(f"x.shape: {x.shape}") 
        # FFT
        x_freq = fft.fftn(x, dim=(-2, -1)) # fft不会改变特征的size
        # x_freq = fft.fftn(x.to('cpu'), dim=(-2, -1)) # 移动到cpu计算
        x_freq = fft.fftshift(x_freq, dim=(-2, -1)) # torch.Size([392, 2048, 4])
        b, dim, T = x_freq.shape
        dim_center = dim //2 
        horizon_center = T //2 
        mask_low = torch.ones((b, dim, T), device=device) 
        # mask_low[..., dim_center - threshold:dim_center + threshold, horizon_center- threshold:horizon_center + threshold] = low_scale # 在指定的区间内进行才会放缩，否则就为1
        mask_low[..., dim_center, horizon_center] = low_scale
        mask_high = torch.ones((b, dim, T), device=device) * high_scale
        mask_high[..., dim_center, horizon_center] = 1
        if x_freq.device != mask_low.device:
            mask_low = mask_low.to(x_freq.device)
            mask_high = mask_high.to(x_freq.device)
        x_freq = x_freq * mask_low * mask_high  # mask 对应论文中频域前面乘的bi
        
        magnitude = torch.abs(x_freq[0])
        # magnitude_real = torch.real(x_freq[0])+1
                # 计算能量
        low_energy = torch.sum((magnitude[dim_center, horizon_center])**2)
        total_energy = torch.sum(magnitude**2)

        # 打印能量占比
        high_energy = total_energy - low_energy 
        print(f"Low Frequency Energy : {low_energy}")
        print(f"High Frequency Energy : {high_energy}")
        print(f"Low Frequency Energy rata: {low_energy / total_energy:.2%}")
        print(f"High Frequency Energy rata: {high_energy / total_energy:.2%}")
        
            # IFFT
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
        
        x_filtered = x_filtered.type(dtype)
        # 重新送回cuda
        # return x_filtered.cuda()
        return x_filtered
    else:
        return x
    
def Fourier_filter_visual(x, threshold, file_path, low_scale, high_scale, visual=False):
    dtype = x.dtype # torch.float32
    device = x.device
    x = x.type(torch.float32)
    # print(f"x.shape: {x.shape}") 
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1)) # fft不会改变特征的size
    # x_freq = fft.fftn(x.to('cpu'), dim=(-2, -1)) # 移动到cpu计算
    x_freq = fft.fftshift(x_freq, dim=(-2, -1)) # torch.Size([392, 2048, 4])
    
    b, dim, T = x_freq.shape
    dim_center = dim //2 
    horizon_center = T //2 
    # whether visulization

    low_energy = None
    total_energy = None
    png_files = [f for f in os.listdir(file_path) if f.endswith('.png')]
    if visual == True and len(png_files) < 14:  # edm:79
        magnitude = torch.abs(x_freq[0])
        # magnitude_real = torch.real(x_freq[0])+1
        magnitude_min = magnitude.min()
        magnitude_max = magnitude.max()
                # 计算能量
        low_energy = torch.sum((magnitude[dim_center, horizon_center])**2)
        total_energy = torch.sum(magnitude**2)

        # 打印能量占比
        high_energy = total_energy - low_energy 
        print(f"Low Frequency Energy : {low_energy}")
        print(f"High Frequency Energy : {high_energy}")
        print(f"total Frequency Energy : {total_energy}")
        # print(f"Low Frequency Energy rata: {low_energy / total_energy:.2%}")
        # print(f"High Frequency Energy rata: {high_energy / total_energy:.2%}")
        normalized_magnitude = torch.log((magnitude - magnitude_min) / (magnitude_max - magnitude_min)+1)
        # print(normalized_magnitude[dim_center - threshold-2:dim_center + threshold+2, horizon_center - threshold-2:horizon_center + threshold+2])
        plt.figure()
        plt.imshow(normalized_magnitude.cpu(), cmap='gray', aspect='auto')
        plt.colorbar(label="Normalized Intensity")
        plt.title(f'{dim}x{T} Tensor as Grayscale Image')
        plt.xlabel(f'Columns ({T} channels)')
        plt.ylabel(f'Rows ({dim} points)')
        # plt.legend()
        filename = os.path.join(file_path, f'feature_dim_{dim}_{len(png_files)+1}.png')
        # plt.savefig(filename)
        plt.clf()
        # plt.savefig('fearture_delta_ddpm.png')
        plt.close() 
    
    # print(f"x_freq.shape: {x_freq.shape}")
    # B, C, H, W = x_freq.shape # 
    mask_low = torch.ones((b, dim, T), device=device) 
    # mask_low[..., dim_center - threshold:dim_center + threshold, horizon_center- threshold:horizon_center + threshold] = low_scale # 在指定的区间内进行才会放缩，否则就为1
    mask_low[..., dim_center, horizon_center] = low_scale
    mask_high = torch.ones((b, dim, T), device=device) * high_scale
    mask_high[..., dim_center, horizon_center] = 1
    if x_freq.device != mask_low.device:
        mask_low = mask_low.to(x_freq.device)
        mask_high = mask_high.to(x_freq.device)
    x_freq = x_freq * mask_low * mask_high  # mask 对应论文中频域前面乘的bi
    # x_freq = x_freq * mask_low
    
    # 对整个特征都进行放缩
    # x_freq = x_freq * scale

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    x_filtered = x_filtered.type(dtype)
    # 重新送回cuda
    # return x_filtered.cuda()
    return x_filtered, low_energy, high_energy,total_energy

def freeu_alpha_caluate(backbone_factor, x):
    x_bar = [] # 存放所有层的x_bar
    alpha_l = (backbone_factor-1)*()
    return None

def visual_chart(feature_delta):
    # tensor转移到cpu上
    cpu_feature_delta = [[tensor.cpu().numpy() for tensor in sublist] for sublist in feature_delta]
    block_name = ["encoder_block_resnet_1","encoder_block_resnet_2","encoder_block_resnet_3","mid_block_1","mid_block_2","decoder_block_resnet_1","decoder_block_resnet_2","final_conv"]
    
    # block_name_en = []
    # block_name_de = []
    # for i in range(len(cpu_feature_delta) // 2):
    #     block_name_en.append(f"encoder_block_{i}")
    #     block_name_de.append(f"decoder_block_{i}")
    # block_name = block_name_en+block_name_de 
        
    plt.figure()
    for i, sub_list in enumerate(cpu_feature_delta):
        # plt.plot(sub_list, label=block_name_list[i])
        plt.plot(sub_list, label=block_name[i])
    # print(len(sub_list))
        
    # plt.ylim(0, 0.3)
    plt.xlabel('denoise time step')
    plt.ylabel('feature_delta')
    plt.legend()
    plt.savefig('fearture_delta_ddim_10.png')
    # plt.savefig('fearture_delta_ddpm.png')
    plt.close()
    
    return None

def register_time(unet, t):
    # print("-----------register_time------------")
    setattr(unet, 'order', t)
    
def register_alpha(unet, coefficient):
    # print("-----------register_time------------")
    setattr(unet, 'alpha', coefficient)
    
def register_faster_forward(model, mod = 'ununi'):
    def faster_forward(self):
        def forward(
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
            """
            x: (B,T,input_dim) [ batch_size,horizon,in_channel ]
            timestep: (B,) or int, diffusion step
            local_cond: (B,T,local_cond_dim)
            global_cond: (B,global_cond_dim)
            output: (B,T,input_dim)
            """
            # print(f"original sample.shape:{sample.shape}")
            sample = einops.rearrange(sample, 'b h t -> b t h')

            # 1. time  针对time的类型进行了修改 原先的unet时间步就只有一步，现在传播会有链式多步  
            if isinstance(timestep, list):
                timesteps = timestep[0]
                step = len(timestep)
            else:
                timesteps = timestep
                step = 1
            if not torch.is_tensor(timesteps) and (not isinstance(timesteps,list)):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps" # 判断sample的设备类型
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64 # 设置timesteps的数据类型
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif (not isinstance(timesteps,list)) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device) # 空集处理
            
            if (not isinstance(timesteps,list)) and len(timesteps.shape) == 1:
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0]) # 如果timesteps是1维的，则扩展到batch维度
            elif isinstance(timesteps, list): # 如果timesteps是list类型
                #timesteps list, such as [981,961,941]
                timesteps = warpped_timestep(timesteps, sample.shape[0]).to(sample.device)  # 对多时间步进行整合分析  把tensor变成list

            global_feature = self.diffusion_step_encoder(timesteps) # 时间嵌入特征

            if global_cond is not None:
                global_feature = torch.cat([
                    global_feature, global_cond
                ], axis=-1)
            
            low_cc, high_cc = noise_compensate_ddim_c()
            # print(f"global_feature.shape:{global_feature.shape}")  # global_feature.shape:torch.Size([56, 260])
            #===============  这个地方是一个修改过的模块 ================   
            order = self.order
            #===============
            # cond = order in [0, 1, 2, 3, 5, 8, 12, 20, 40, 70]
            if isinstance(mod, int): # 不同的key时间步模式设置
                    cond = order % mod == 0
            elif mod == "ununi":
                cond = order in [0,8,9] # [0, 7, 9] [0,1,2,3,4,5,6,7,8,9] [0, 5, 7 , 8, 9]
            elif mod == "uni":
                cond = order in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
                
            # encode local features
            h_local = list()
            if local_cond is not None:  # 先经过resnet转化为对应的特征
                local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
                resnet, resnet2 = self.local_cond_encoder
                x = resnet(local_cond, global_feature)
                h_local.append(x)
                x = resnet2(local_cond, global_feature)
                h_local.append(x)
                
            if cond: 
                # 如果是key step，则运行传播
                x = sample
                bs = x.shape[0]
                # print(f"x.shape:{x.shape}")
                h = []
                down_block_res_samples_list = (x,)
                # print(type(down_block_res_samples_list))
                # down encoder
                skip_feature = ()
                for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
                    x = resnet(x, global_feature) # 时间步作为全局特征？
                    if idx == 0 and len(h_local) > 0:
                        x = x + h_local[0]
                    x = resnet2(x, global_feature) # 先经过两个resnet 得到特征先存储在h中 对应sample
                    # print(type(x))
                    # h.append(x)   
                    skip_feature += (x,)   
                    x = downsample(x) # re sample 经过下采样得到得特征
                    
                    
                # ----------------在这个地方对特征进行存储，用来传播---------#

                for mid_module in self.mid_modules:
                    x = mid_module(x, global_feature)
                    
                #----------------------save feature-------------------------
                # setattr(self, 'skip_feature', (tmp_sample.clone() for tmp_sample in down_block_res_samples))
                # 这个地方相当于是直接把res 特征直接复制作为当前的skip_feature 但是这也不对啊，现在是key time step，这个地方的特征应该用的是自己的
                setattr(self, 'skip_feature', deepcopy(skip_feature)) # 将unet的变量skip_feature值为down_block_res_samples的深拷贝 为了不影响当前key step的特征
                setattr(self, 'toup_feature', x.detach().clone())    # 就是经过mid输出的结果
                # setattr(self, 'global_feature', deepcopy(global_feature))    # 就是经过mid输出的结果
                #-----------------------save feature------------------------



                #-------------------expand feature for parallel---------------
                if isinstance(timestep, list): # 又重新进行一遍？  timesteps包含了当前key step 所传播到的non key step  在最开始是单步的，所以不执行，只有多步有影响才执行下方代码
                    #timesteps list, such as [981,961,941]
                    timesteps = warpped_timestep(timestep, x.shape[0]).to(x.device) 
                    # print(f"time step:{timesteps}")
                    # global_feature = self.diffusion_step_encoder(timesteps)  # global_feature.shape:torch.Size([56, 128])
                    
                # print(f"step:{step}")  
                # 如何作用到后面的非key step？ 相当于直接进行跳步了，例如key time 15 的特征直接输入 最后一个替换特征时间步24的unet的 decoder （但是为什么不是先经过 15 步的mid，然后再把特征给24，或者15的down特征输到24 的mid进行编码？
                skip_feature = warpped_skip_feature(skip_feature, step) # 既然是跳步 那这个部分有啥用？ 直接把原先的特征映射到 step=24时候的模块输入了
                x = warpped_feature(x, step)
                global_feature = warpped_global_feature(global_feature, step)
                #-------------------expand feature for parallel---------------

            else:
                print("--------------cond=false------------------") # 这里都没运行是为啥？？？？  只有一个cond = true
                skip_feature = self.skip_feature # else里面没有直接定义啊？ 直接沿用之前的了 就是在非key的时候直接沿用最近一次key的feature
                x = self.toup_feature 

                #-------------------expand feature for parallel---------------
                print(f"non cond step:{step}")  # 为什么不与运行？？
                skip_feature = warpped_skip_feature(skip_feature, step) # step 反应的是当前一致encoder特征的时间步数
                x = warpped_feature(x, step)
                #-------------------expand feature for parallel---------------

             # up decoder
            # pdb.set_trace()
            # x = x*0
            for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
                # x = x[-bs:].squeeze(0) # 传播之后的特征 此时h也进行拓展，所以是n time step gap 组特征
                # x = torch.cat((x, h.pop()), dim=1)
                skip_feature = list(skip_feature)
                s_feature = skip_feature.pop()
                # idx 1 = encoder block2 idx 0 = encoder block3
                # if idx == 0 or idx == 1:
                #     s_feature = s_feature*0
                
                # =====================在此处进行freeu backbone_feature 增强=====================
                # 原始论文方法是只对最里面的两个resblock进行处理；对于dp的1d unet，只有两个upblock结合了skip_feature，所以没有影响
                # if x.shape[1] == 2048:
                #     x[:,:1024] = x[:,:1024] * self.b1 # 为什么不是根据结构来计算的？
                #     s_feature  = Fourier_filter(s_feature , threshold=1, scale=self.s1) # 仅仅是对特征输入进行修改了
                # if x.shape[1] == 1024:
                #     x[:,:512] = x[:,:512] * self.b2
                #     s_feature  = Fourier_filter(s_feature , threshold=1, scale=self.s2)
                # ===============================================================================
                
                # print(f"s_feature shape:{s_feature.shape}")
                # print(f"s_feature type:{type(s_feature)}")
                
                x = torch.cat((x, s_feature), dim=1)
                skip_feature = tuple(skip_feature) 
                
                # 如果在前面先运行一遍.pop(),就会导致此处再运行的h就不一样了，从而导致报错。.pop() 移除最后一个元素 相当于移除了encoder的最后一个特征  为什么还保留了encoder前n-1层的特征作为输入？ 但是这个h貌似不影响，因为此时的h是key step的特征
                x = resnet(x, global_feature)
                # The correct condition should be:
                # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
                # However this change will break compatibility with published checkpoints.
                # Therefore it is left as a comment. 
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
                
                # pdb.set_trace()
                # if reuse，noise compensate 跳过第一步
                if x.shape[1] == 512:
                    num_nonkey = x.shape[0] / bs
                    if num_nonkey > 1:
                        # noise_feature = x[0:bs]
                        l = 1
                        h = 1
                        for step in range(1,int(num_nonkey)): 
                            low_scale = low_cc[order+step]
                            high_scale = high_cc[order+step]
                            # l = l*low_scale
                            # h = h*high_scale
                            print("======================================================================")
                            x[bs*step:bs*(step+1)] = Fourier_filter(x[bs*step:bs*(step+1)], low_scale=l, high_scale=h, flag=True)
                x = upsample(x)
                
            # 6. post-process
            x = self.final_conv(x)  # 最终输出经过一个卷积层，但是输入时候为什么没有卷积层进行处理？

            x = einops.rearrange(x, 'b t h -> b h t')
            return x
    
        # 把encoder propagation 的forward迁移过来
        return forward
    if model.__class__.__name__ == 'ConditionalUnet1D':
        model.forward = faster_forward(model)
        # setattr(model, 'b1', b1)
        # setattr(model, 'b2', b2)
        # setattr(model, 's1', s1)
        # setattr(model, 's2', s2) 
        
def register_normal_forward(model, mod = 'ununi'):
    def faster_forward(self):
        def forward( 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
            """
            x: (B,T,input_dim)
            timestep: (B,) or int, diffusion step
            local_cond: (B,T,local_cond_dim)
            global_cond: (B,global_cond_dim)
            output: (B,T,input_dim)
            """
            sample = einops.rearrange(sample, 'b h t -> b t h')

            # 1. time
            timesteps = timestep
            if not torch.is_tensor(timesteps): # 从stabele diffusion里面扒的
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])

            global_feature = self.diffusion_step_encoder(timesteps)

            if global_cond is not None:
                global_feature = torch.cat([
                    global_feature, global_cond
                ], axis=-1)
            
            # encode local features
            h_local = list()
            if local_cond is not None:
                local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
                resnet, resnet2 = self.local_cond_encoder
                x = resnet(local_cond, global_feature)
                h_local.append(x)
                x = resnet2(local_cond, global_feature)
                h_local.append(x)
            
            x = sample
            h = []
            for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
                x = resnet(x, global_feature) # 时间步作为全局特征？
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
                h.append(x)
                x = downsample(x) # 
                
            # ----------------在这个地方对特征进行存储，用来传播---------#

            for mid_module in self.mid_modules:
                x = mid_module(x, global_feature)

            for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
                x = torch.cat((x, h.pop()), dim=1)
                x = resnet(x, global_feature)
                # The correct condition should be:
                # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
                # However this change will break compatibility with published checkpoints.
                # Therefore it is left as a comment.
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
                x = upsample(x)

            x = self.final_conv(x)

            x = einops.rearrange(x, 'b t h -> b h t')
            return x
    
        # 把encoder propagation 的forward迁移过来
        return forward
    if model.__class__.__name__ == 'ConditionalUnet1D':
        model.forward = faster_forward(model)
        # setattr(model, 'b1', b1)
        # setattr(model, 'b2', b2)
        # setattr(model, 's1', s1)
        # setattr(model, 's2', s2) 
        
# normal dp + free
def freeu_register_normal_forward(model, e1=1 ,e2=1, b=1):
    def faster_forward(self):
        def forward( 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, flag = False, **kwargs):
            """
            x: (B,T,input_dim)
            timestep: (B,) or int, diffusion step
            local_cond: (B,T,local_cond_dim)
            global_cond: (B,global_cond_dim)
            output: (B,T,input_dim)
            """
            sample = einops.rearrange(sample, 'b h t -> b t h')

            # 1. time
            timesteps = timestep
            if not torch.is_tensor(timesteps): # 从stabele diffusion里面扒的
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])

            global_feature = self.diffusion_step_encoder(timesteps)

            if global_cond is not None:
                global_feature = torch.cat([
                    global_feature, global_cond
                ], axis=-1)
            
            # encode local features
            h_local = list()
            if local_cond is not None:
                local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
                resnet, resnet2 = self.local_cond_encoder
                x = resnet(local_cond, global_feature)
                h_local.append(x)
                x = resnet2(local_cond, global_feature)
                h_local.append(x)
            
            x = sample
            h = []
            for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
                x = resnet(x, global_feature) # 时间步作为全局特征？
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
                h.append(x)
                x = downsample(x) # 
                
            # ----------------在这个地方对特征进行存储，用来传播---------#

            for mid_module in self.mid_modules:
                x = mid_module(x, global_feature)

            # file_path = './data/fft_fig/block'
            # os.makedirs(file_path,  exist_ok=True)

            for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
                s_feature = h.pop()
                
                # =====================在此处进行freeu backbone_feature 增强=====================
                if x.shape[1] == 2048:
                    x = x * self.backbone_scale
                    # x[:,:1024] = x[:,:1024] * self.low_scale # 为什么不是根据结构来计算的？
                    # s_feature  = Fourier_filter(s_feature , threshold=1, low_scale=self.low_scale, high_scale=self.high_scale, visual=True) # 仅仅是对特征输入进行修改了
                    s_feature = s_feature * self.encoder_2_scale
                if x.shape[1] == 1024:
                    # x = x * self.low_scale
                #     # x[:,:512] = x[:,:512] * self.high_scale
                    # s_feature  = Fourier_filter(s_feature, threshold=1, file_path=None,low_scale=1, high_scale=1, visual=False)
                    s_feature = s_feature * self.encoder_1_scale
                # ===============================================================================
                    
                x = torch.cat((x, s_feature), dim=1) 
                # if x.shape[1] == 2048: # decoder block2  input 2048 4096
                #         _ = Fourier_filter(x , threshold=1, low_scale=1, high_scale=1, visual=True) # 仅仅是对特征输入进行修改了
                x = resnet(x, global_feature)
                # The correct condition should be:
                # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
                # However this change will break compatibility with published checkpoints.
                # Therefore it is left as a comment.
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
                
                # =====================在此处进行freeu backbone_feature 增强=====================
                if x.shape[1] == 512: # 直接在decoder block2 进行增强 
                        file_path = './data/fft_fig'
                        os.makedirs(file_path,  exist_ok=True)
                        _, low_energy, high_energy = Fourier_filter_visual(x , threshold=1, file_path=file_path,low_scale=1, high_scale=1, visual=True) # 仅仅是对特征输入进行修改了
                    # s_feature = s_feature * self.s1 
                # ===============================================================================
                # 只对最后一次去噪的result进行修改
                        # pdb.set_trace()
                        print(flag)
                        x = Fourier_filter(x, low_scale=1, high_scale=1, flag=flag)
                x = upsample(x)

            x = self.final_conv(x)

            x = einops.rearrange(x, 'b t h -> b h t')
            return x, low_energy, high_energy
    
        # 把encoder propagation 的forward迁移过来
        return forward
    if model.__class__.__name__ == 'ConditionalUnet1D':
        model.forward = faster_forward(model)
        setattr(model, 'encoder_1_scale', e1)
        setattr(model, 'encoder_2_scale', e2)
        setattr(model, 'backbone_scale', b)
        # setattr(model, 's2', s2) 

def register_edm_faster_forward(model, mod = 'ununi'):
    def faster_forward(self):
        def forward(
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
            """
            x: (B,T,input_dim) [ batch_size,horizon,in_channel ]
            timestep: (B,) or int, diffusion step
            local_cond: (B,T,local_cond_dim)
            global_cond: (B,global_cond_dim)
            output: (B,T,input_dim)
            """
            # print(f"original sample.shape:{sample.shape}")
            sample = einops.rearrange(sample, 'b h t -> b t h')

            # 1. time  针对time的类型进行了修改 原先的unet时间步就只有一步，现在传播会有链式多步  
            if isinstance(timestep, list):
                timesteps = timestep[0]
                step = len(timestep)
            else:
                timesteps = timestep
                step = 1
            if not torch.is_tensor(timesteps) and (not isinstance(timesteps,list)):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps" # 判断sample的设备类型
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64 # 设置timesteps的数据类型
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif (not isinstance(timesteps,list)) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device) # 空集处理
            
            if (not isinstance(timesteps,list)) and len(timesteps.shape) == 1:
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0]) # 如果timesteps是1维的，则扩展到batch维度
            elif isinstance(timesteps, list): # 如果timesteps是list类型
                #timesteps list, such as [981,961,941]
                timesteps = warpped_timestep(timesteps, sample.shape[0]).to(sample.device)  # 对多时间步进行整合分析  把tensor变成list

            global_feature = self.diffusion_step_encoder(timesteps) # 时间嵌入特征 # [1,402]

            if global_cond is not None:
                global_feature = torch.cat([
                    global_feature, global_cond
                ], axis=-1)
            
            # print(f"global_feature.shape:{global_feature.shape}")  # global_feature.shape:torch.Size([56, 260])
            #===============  这个地方是一个修改过的模块 ================   
            order = self.order # 此处的order依旧就timestep不是times，所以可以离散的step进行确定是否为关键步
            #===============
            # cond = order in [0, 1, 2, 3, 5, 8, 12, 20, 40, 70]
            if isinstance(mod, int): # 不同的key时间步模式设置
                    cond = order % mod == 0
            # [0, 7, 9] [0,1,2,3,4,5,6,7,8,9] [0, 5, 7 , 8, 9]
            elif mod == "ununi":
                cond = order in list(range(0,80,3))
            elif mod == "uni":
                cond = order in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
                
            # encode local features
            h_local = list()
            if local_cond is not None:  # 先经过resnet转化为对应的特征
                local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
                resnet, resnet2 = self.local_cond_encoder
                x = resnet(local_cond, global_feature)
                h_local.append(x)
                x = resnet2(local_cond, global_feature)
                h_local.append(x)
                
            if cond: 
                # 如果是key step，则运行传播
                x = sample
                h = []
                # print(type(down_block_res_samples_list))
                # down encoder
                skip_feature = ()
                for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
                    x = resnet(x, global_feature) # 时间步作为全局特征？
                    if idx == 0 and len(h_local) > 0:
                        x = x + h_local[0]
                    x = resnet2(x, global_feature) # 先经过两个resnet 得到特征先存储在h中 对应sample   
                    skip_feature += (x,)   
                    x = downsample(x) # re sample 经过下采样得到得特征 
                         
                # ----------------在这个地方对特征进行存储，用来传播---------#

                for mid_module in self.mid_modules:
                    x = mid_module(x, global_feature)
                    
                #----------------------save feature-------------------------
                # setattr(self, 'skip_feature', (tmp_sample.clone() for tmp_sample in down_block_res_samples))
                # 这个地方相当于是直接把res 特征直接复制作为当前的skip_feature 但是这也不对啊，现在是key time step，这个地方的特征应该用的是自己的
                setattr(self, 'skip_feature', deepcopy(skip_feature)) # 将unet的变量skip_feature值为down_block_res_samples的深拷贝 为了不影响当前key step的特征
                setattr(self, 'toup_feature', x.detach().clone())    # 就是经过mid输出的结果
                # setattr(self, 'global_feature', deepcopy(global_feature))    # 就是经过mid输出的结果
                #-----------------------save feature------------------------



                #-------------------expand feature for parallel---------------
                if isinstance(timestep, list): # 又重新进行一遍？  timesteps包含了当前key step 所传播到的non key step  在最开始是单步的，所以不执行，只有多步有影响才执行下方代码
                    #timesteps list, such as [981,961,941]
                    timesteps = warpped_timestep(timestep, x.shape[0]).to(x.device) 
                    # print(f"time step:{timesteps}")
                    # global_feature = self.diffusion_step_encoder(timesteps)  # global_feature.shape:torch.Size([56, 128])
                    
                # print(f"step:{step}")  
                # 如何作用到后面的非key step？ 相当于直接进行跳步了，例如key time 15 的特征直接输入 最后一个替换特征时间步24的unet的 decoder （但是为什么不是先经过 15 步的mid，然后再把特征给24，或者15的down特征输到24 的mid进行编码？
                skip_feature = warpped_skip_feature(skip_feature, step) # 既然是跳步 那这个部分有啥用？ 直接把原先的特征映射到 step=24时候的模块输入了
                x = warpped_feature(x, step)
                global_feature = warpped_global_feature(global_feature, step)
                #-------------------expand feature for parallel---------------

            else:
                print("--------------cond=false------------------") # 这里都没运行是为啥？？？？  只有一个cond = true
                skip_feature = self.skip_feature # else里面没有直接定义啊？ 直接沿用之前的了 就是在非key的时候直接沿用最近一次key的feature
                x = self.toup_feature 

                #-------------------expand feature for parallel---------------
                print(f"non cond step:{step}")  # 为什么不与运行？？
                skip_feature = warpped_skip_feature(skip_feature, step) # step 反应的是当前一致encoder特征的时间步数
                x = warpped_feature(x, step)
                #-------------------expand feature for parallel---------------

            # up decoder
            # pdb.set_trace()
            for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
                # x = x[-bs:].squeeze(0) # 传播之后的特征 此时h也进行拓展，所以是n time step gap 组特征
                # x = torch.cat((x, h.pop()), dim=1)
                skip_feature = list(skip_feature)
                s_feature = skip_feature.pop()
                # idx 1 = encoder block2 idx 0 = encoder block3
                x = torch.cat((x, s_feature), dim=1)
                skip_feature = tuple(skip_feature) 
                
                # 如果在前面先运行一遍.pop(),就会导致此处再运行的h就不一样了，从而导致报错。.pop() 移除最后一个元素 相当于移除了encoder的最后一个特征  为什么还保留了encoder前n-1层的特征作为输入？ 但是这个h貌似不影响，因为此时的h是key step的特征
                x = resnet(x, global_feature)
                # The correct condition should be:
                # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
                # However this change will break compatibility with published checkpoints.
                # Therefore it is left as a comment. 
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
                x = upsample(x)
                
            # 6. post-process
            x = self.final_conv(x)  # 最终输出经过一个卷积层，但是输入时候为什么没有卷积层进行处理？

            x = einops.rearrange(x, 'b t h -> b h t')
            return x
    
        return forward
    if model.__class__.__name__ == 'ConditionalUnet1D':
        model.forward = faster_forward(model)
        
def register_faster_heun_solver(scheduler):
    def faster_heun_solver(self):
        def heun_solver(
            model, 
            samples, 
            t, 
            next_t,  
            clamp = False
        ):
            dims = samples.ndim
            y = samples
            # pdb.set_trace()
            denoisedy = self.calc_out(model, y, t, clamp = clamp) # 得到一组噪声
            bs = denoisedy.shape[0]
            if not isinstance (t, list):
                t = [t]
                next_t = [next_t]
            bs_perstep = bs//len(t)
            
            y_next_record = []
            denoisedy_next = None
            curr_y = copy.deepcopy(y)
            
            for idx, (times, next_times) in enumerate(zip(t, next_t)):
                step = append_dims((next_times - times), dims) # 维度对齐 此时的得到得一组list
                noise_y = denoisedy[idx*bs_perstep:(idx+1)*bs_perstep]
                # dy = (y - denoisedy) / append_dims(t, dims)
                dy = (curr_y - noise_y) / append_dims(times, dims)
                
                y_next = samples + step * dy
                # 将第一次的y_next的噪声reuse，和encoder reuse本质上没有区别 每一次reuse最后一步重新计算提高准确率 10+20/80+80
                # if denoisedy_next is None or idx == len(t) - 1:
                #     denoisedy_next = self.calc_out(model, y_next, next_times, clamp = clamp) 
                # else:
                #     continue
                denoisedy_next = self.calc_out(model, y_next, next_times, clamp = clamp) 
                dy_next = (y_next - denoisedy_next) / append_dims(next_times, dims)
                y_next = samples + step * (dy + dy_next) / 2
                
                curr_y = copy.deepcopy(y_next)
                y_next_record.append(y_next)
                
                  
            return y_next
        
        return heun_solver
    if scheduler.__class__.__name__ == 'Karras_Scheduler':
        scheduler.heun_solver = faster_heun_solver(scheduler)
        
def register_normal_heun_solver(scheduler):
    def faster_heun_solver(self):
        def heun_solver(
            model, 
            samples, 
            t, 
            next_t,  
            clamp = False
        ):
            dims = samples.ndim
            y = samples
            step = append_dims((next_t - t), dims) # 维度对齐
            
            denoisedy = self.calc_out(model, y, t, clamp = clamp)
            dy = (y - denoisedy) / append_dims(t, dims)


            y_next = samples + step * dy 

            denoisedy_next = self.calc_out(model, y_next, next_t, clamp = clamp)
            dy_next = (y_next - denoisedy_next) / append_dims(next_t, dims)

            y_next = samples + step * (dy + dy_next) / 2



            return y_next
        
        return heun_solver
    if scheduler.__class__.__name__ == 'Karras_Scheduler':
        scheduler.heun_solver = faster_heun_solver(scheduler)