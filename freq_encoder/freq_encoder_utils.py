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
    semi_bs = bs//2
    ts = []
    for timestep in timesteps:
        timestep = timestep[None]
        texp = timestep.expand(semi_bs)
        ts.append(texp)
    timesteps = torch.cat(ts)
    return timesteps.repeat(2,1).reshape(-1)

def Fourier_filter(x, low_scale, high_scale, flag=False):
    if flag == True:
        dtype = x.dtype # torch.float32
        device = x.device
        x = x.type(torch.float32)

        # FFT
        x_freq = fft.fftn(x, dim=(-2, -1)) 
        x_freq = fft.fftshift(x_freq, dim=(-2, -1)) # torch.Size([392, 2048, 4])
        b, dim, T = x_freq.shape
        dim_center = dim //2 
        horizon_center = T //2 
        mask_low = torch.ones((b, dim, T), device=device) 
        # mask_low[..., dim_center - threshold:dim_center + threshold, horizon_center- threshold:horizon_center + threshold] = low_scale 
        mask_low[..., dim_center, horizon_center] = low_scale
        mask_high = torch.ones((b, dim, T), device=device) * high_scale
        mask_high[..., dim_center, horizon_center] = 1
        if x_freq.device != mask_low.device:
            mask_low = mask_low.to(x_freq.device)
            mask_high = mask_high.to(x_freq.device)
        x_freq = x_freq * mask_low * mask_high  
        
        magnitude = torch.abs(x_freq[0])

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

        return x_filtered
    else:
        return x
    
def Fourier_filter_visual(x, threshold, file_path, low_scale, high_scale, visual=False):
    dtype = x.dtype # torch.float32
    device = x.device
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1)) 
    x_freq = fft.fftshift(x_freq, dim=(-2, -1)) # torch.Size([392, 2048, 4])
    
    b, dim, T = x_freq.shape
    dim_center = dim //2 
    horizon_center = T //2 
    # whether visulization
    png_files = [f for f in os.listdir(file_path) if f.endswith('.png')]
    if visual == True and len(png_files) < 16:  # edm:79
        magnitude = torch.abs(x_freq[0])
        # magnitude_real = torch.real(x_freq[0])+1
        magnitude_min = magnitude.min()
        magnitude_max = magnitude.max()

        low_energy = torch.sum((magnitude[dim_center, horizon_center])**2)
        total_energy = torch.sum(magnitude**2)

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
        plt.savefig(filename)
        plt.clf()
        # plt.savefig('fearture_delta_ddpm.png')
        plt.close() 
    

    # B, C, H, W = x_freq.shape 
    mask_low = torch.ones((b, dim, T), device=device) 
    # mask_low[..., dim_center - threshold:dim_center + threshold, horizon_center- threshold:horizon_center + threshold] = low_scale 
    mask_low[..., dim_center, horizon_center] = low_scale
    mask_high = torch.ones((b, dim, T), device=device) * high_scale
    mask_high[..., dim_center, horizon_center] = 1
    if x_freq.device != mask_low.device:
        mask_low = mask_low.to(x_freq.device)
        mask_high = mask_high.to(x_freq.device)
    x_freq = x_freq * mask_low * mask_high  
    
    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    x_filtered = x_filtered.type(dtype)

    return x_filtered, low_energy, high_energy,total_energy

def visual_chart(feature_delta):
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
    
    
def register_faster_encoder_forward(model, mod = 'ununi'):
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
            sample = einops.rearrange(sample, 'b h t -> b t h')

            # 1. time  The original unet time step was only one step, but now the propagation will have multiple steps in a chain
            if isinstance(timestep, list):
                timesteps = timestep[0]
                step = len(timestep)
            else:
                timesteps = timestep
                step = 1
            if not torch.is_tensor(timesteps) and (not isinstance(timesteps,list)):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps" 
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64 
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif (not isinstance(timesteps,list)) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device) 
            
            if (not isinstance(timesteps,list)) and len(timesteps.shape) == 1:
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0]) 
            elif isinstance(timesteps, list):
                #timesteps list, such as [981,961,941]
                timesteps = warpped_timestep(timesteps, sample.shape[0]).to(sample.device)  

            global_feature = self.diffusion_step_encoder(timesteps) 

            if global_cond is not None:
                global_feature = torch.cat([
                    global_feature, global_cond
                ], axis=-1)
            
            cc = noise_compensate_ddim_c(task = "transport_mh", scheduler="DDIM")
 
            order = self.order
            if isinstance(mod, int): # Different key time step mode settings
                    cond = order % mod == 0
            elif mod == "ununi":
                cond = order in [0, 8, 9] 
            elif mod == "uni":
                cond = order in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
                
            # encode local features
            h_local = list()
            if local_cond is not None:  
                local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
                resnet, resnet2 = self.local_cond_encoder
                x = resnet(local_cond, global_feature)
                h_local.append(x)
                x = resnet2(local_cond, global_feature)
                h_local.append(x)
                
            if cond: 
                # If it is a key step, run propagation
                x = sample
                bs = x.shape[0]
                # down encoder
                skip_feature = ()
                for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
                    x = resnet(x, global_feature) 
                    if idx == 0 and len(h_local) > 0:
                        x = x + h_local[0]
                    x = resnet2(x, global_feature) 

                    skip_feature += (x,)   
                    x = downsample(x)                

                for mid_module in self.mid_modules:
                    x = mid_module(x, global_feature)
                    
                #----------------------save feature-------------------------
                # setattr(self, 'skip_feature', (tmp_sample.clone() for tmp_sample in down_block_res_samples))
                setattr(self, 'skip_feature', deepcopy(skip_feature))
                setattr(self, 'toup_feature', x.detach().clone())   
                # setattr(self, 'global_feature', deepcopy(global_feature))
                #-----------------------save feature------------------------

                #-------------------expand feature for parallel---------------
                if isinstance(timestep, list): 
                    #timesteps list, such as [981,961,941]
                    timesteps = warpped_timestep(timestep, x.shape[0]).to(x.device) 
                    # global_feature = self.diffusion_step_encoder(timesteps)  # global_feature.shape:torch.Size([56, 128])
                skip_feature = warpped_skip_feature(skip_feature, step) 
                x = warpped_feature(x, step)
                global_feature = warpped_global_feature(global_feature, step)
                #-------------------expand feature for parallel---------------

            else:
                print("--------------cond=false------------------") 
                skip_feature = self.skip_feature 
                x = self.toup_feature 

                #-------------------expand feature for parallel---------------
                print(f"non cond step:{step}")  
                skip_feature = warpped_skip_feature(skip_feature, step) 
                x = warpped_feature(x, step)
                #-------------------expand feature for parallel---------------

            # up decoder
            for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
                # x = x[-bs:].squeeze(0) 
                # x = torch.cat((x, h.pop()), dim=1)
                skip_feature = list(skip_feature)
                s_feature = skip_feature.pop()
                
                x = torch.cat((x, s_feature), dim=1)
                skip_feature = tuple(skip_feature) 
                x = resnet(x, global_feature)
                # The correct condition should be:
                # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
                # However this change will break compatibility with published checkpoints.
                # Therefore it is left as a comment. 
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
                
                # # If you want noise compensation, uncomment this section                
                # if x.shape[1] == 512:
                #     num_nonkey = x.shape[0] / bs
                #     if num_nonkey > 1:
                #         for step in range(1,int(num_nonkey)): 
                #             x[bs*step:bs*(step+1)] = x[bs*step:bs*(step+1)]*cc[order+step]
                            
                x = upsample(x)

            # 6. post-process
            x = self.final_conv(x) 

            x = einops.rearrange(x, 'b t h -> b h t')
            return x
    
        return forward
    if model.__class__.__name__ == 'ConditionalUnet1D':
        model.forward = faster_forward(model)


        
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
            if not torch.is_tensor(timesteps):
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
                x = resnet(x, global_feature) 
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
                h.append(x)
                x = downsample(x) # 

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

            # 1. time   
            if isinstance(timestep, list):
                timesteps = timestep[0]
                step = len(timestep)
            else:
                timesteps = timestep
                step = 1
            if not torch.is_tensor(timesteps) and (not isinstance(timesteps,list)):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps" 
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64 
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif (not isinstance(timesteps,list)) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device) 
            
            if (not isinstance(timesteps,list)) and len(timesteps.shape) == 1:
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0]) 
            elif isinstance(timesteps, list): 
                #timesteps list, such as [981,961,941]
                timesteps = warpped_timestep(timesteps, sample.shape[0]).to(sample.device)  

            global_feature = self.diffusion_step_encoder(timesteps) 

            if global_cond is not None:
                global_feature = torch.cat([
                    global_feature, global_cond
                ], axis=-1)
            
            cc = noise_compensate_ddim_c(task = "tool_hang_ph", scheduler="EDM")
            order = self.order 
            if isinstance(mod, int): 
                    cond = order % mod == 0
            elif mod == "ununi":
                cond = order in [0, 5, 6]
            elif mod == "uni":
                cond = order in [0,2,4]
                
            # encode local features
            h_local = list()
            if local_cond is not None:  
                local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
                resnet, resnet2 = self.local_cond_encoder
                x = resnet(local_cond, global_feature)
                h_local.append(x)
                x = resnet2(local_cond, global_feature)
                h_local.append(x)
                
            if cond: 
                x = sample
                h = []
                bs = x.shape[0]
                # down encoder
                skip_feature = ()
                for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
                    x = resnet(x, global_feature) 
                    if idx == 0 and len(h_local) > 0:
                        x = x + h_local[0]
                    x = resnet2(x, global_feature)    
                    skip_feature += (x,)   
                    x = downsample(x) 

                for mid_module in self.mid_modules:
                    x = mid_module(x, global_feature)

                #----------------------save feature-------------------------
                # setattr(self, 'skip_feature', (tmp_sample.clone() for tmp_sample in down_block_res_samples))
                setattr(self, 'skip_feature', deepcopy(skip_feature)) 
                setattr(self, 'toup_feature', x.detach().clone())   
                # setattr(self, 'global_feature', deepcopy(global_feature))
                #-----------------------save feature------------------------

                #-------------------expand feature for parallel---------------
                if isinstance(timestep, list): 
                    #timesteps list, such as [981,961,941]
                    timesteps = warpped_timestep(timestep, x.shape[0]).to(x.device) 
                    # global_feature = self.diffusion_step_encoder(timesteps)  # global_feature.shape:torch.Size([56, 128])

                skip_feature = warpped_skip_feature(skip_feature, step) 
                x = warpped_feature(x, step)
                global_feature = warpped_global_feature(global_feature, step)
                #-------------------expand feature for parallel---------------

            else:
                print("--------------cond=false------------------") 
                skip_feature = self.skip_feature 
                x = self.toup_feature 
                #-------------------expand feature for parallel---------------
                print(f"non cond step:{step}")  
                skip_feature = warpped_skip_feature(skip_feature, step)
                x = warpped_feature(x, step)
                #-------------------expand feature for parallel---------------

            # up decoder
            for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
                # x = x[-bs:].squeeze(0) 
                # x = torch.cat((x, h.pop()), dim=1)
                skip_feature = list(skip_feature)
                s_feature = skip_feature.pop()
                # idx 1 = encoder block2 idx 0 = encoder block3
                x = torch.cat((x, s_feature), dim=1)
                skip_feature = tuple(skip_feature) 
                
                x = resnet(x, global_feature)
                # The correct condition should be:
                # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
                # However this change will break compatibility with published checkpoints.
                # Therefore it is left as a comment. 
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)

                # # If you want noise compensation, uncomment this section
                # if x.shape[1] == 512:
                #     num_nonkey = x.shape[0] / bs
                #     if num_nonkey > 1:
                #         for step in range(1,int(num_nonkey)): 
                #             x[bs*step:bs*(step+1)] = x[bs*step:bs*(step+1)]*cc[order+step]
    
                x = upsample(x)
                
            # 6. post-process
            x = self.final_conv(x)

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
            denoisedy = self.calc_out(model, y, t, clamp = clamp) 
            bs = denoisedy.shape[0]
            if not isinstance (t, list):
                t = [t]
                next_t = [next_t] 
            bs_perstep = bs//len(t)
            
            denoisedy_next = None
            curr_y = copy.deepcopy(y)
            curr_traj = copy.deepcopy(samples)
            for idx, (times, next_times) in enumerate(zip(t, next_t)):
                step = append_dims((next_times - times), dims)
                noise_y = denoisedy[idx*bs_perstep:(idx+1)*bs_perstep]
                # dy = (y - denoisedy) / append_dims(t, dims)
                dy = (curr_y - noise_y) / append_dims(times, dims)
                
                y_next = curr_traj + step * dy
                if denoisedy_next is None or idx == len(t) - 1:
                    denoisedy_next = self.calc_out(model, y_next, next_times, clamp = clamp) 

                dy_next = (y_next - denoisedy_next) / append_dims(next_times, dims)
                y_next = curr_traj + step * (dy + dy_next) / 2
                
                curr_y = copy.deepcopy(y_next)
                curr_traj = copy.deepcopy(y_next)   
                  
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
            step = append_dims((next_t - t), dims) 
            
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