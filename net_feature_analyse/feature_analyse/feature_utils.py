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
import imageio
import datetime
import PIL.Image
import einops
import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import torch.fft as fft
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from copy import deepcopy
from tqdm import tqdm
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, Tuple, List, Callable, Dict, Any
from scipy.ndimage import gaussian_filter1d

def mse_caluate(feature):
    if isinstance(feature, list):
        if len(feature) < 2:
            raise ValueError("Feature list must contain at least two elements.")
        f1 = feature[1:] 
        f2 = feature[:-1]
    else:
        raise ValueError("feature should be a list")

    assert len(f1) == len(f2)
    mse_loss = []
    for (feat1, feat2) in zip(f1, f2):
        mse_loss.append(F.mse_loss(feat1, feat2)/feat1.shape[0]) 
    # mse_loss = F.mse_loss(f1, f2)

    
    return mse_loss

def get_key_steps(Mse, key_step_num=0, coefficients=None):
    if isinstance(Mse[0], list) and isinstance(Mse[0][0], torch.Tensor):    
        Mse_cpu = [[tensor.cpu().numpy() for tensor in sublist] for sublist in Mse]
    length = len(Mse[0])
    coefficients_array = np.array([
        [coefficients[0]] * length, 
        [coefficients[1]] * length, 
        [coefficients[2]] * length  
    ])
    block_mse = np.array([Mse_cpu[1], Mse_cpu[2], Mse_cpu[4]])
    score = (coefficients_array * block_mse).sum(axis=0)
    
    top_indices = np.argpartition(-score, key_step_num-1)[:key_step_num-1]  
    top_indices = np.sort(top_indices)
    key_steps = [0] 
    key_steps.extend(top_indices+1) 
    if len(key_steps) != key_step_num:
        raise Exception("key steps number is wrong!")
    print("key steps:", key_steps)
    print("final score:", score[top_indices].sum())
    return None

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
        if i < 5: # only encoder and mid blocks
            plt.plot(sub_list, label=block_name[i])
     
    # plt.ylim(0, 0.1)
    plt.xlabel('denoise time step')
    plt.ylabel('feature_delta')
    plt.legend()
    
    now = datetime.datetime.now() 
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    taks_name = 'tool_hang' 
    save_dir = f'./data/{taks_name}_eval_output/feature/' 
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(f'{save_dir}{timestamp}_fearture_delta_edm.png')
    # plt.savefig('fearture_delta_ddpm.png')
    plt.close()
    
    return None


@torch.no_grad()
def feature_visual_block(trajectory, idx, save_dir, type=None):
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    colors = np.linspace(0, 1, len(x))
    plt.figure()
    # plot scatter or both
    if type == 'plot':
        plt.grid(True) 
        plt.plot(x, y, marker='o', color='r', linestyle='-', label='Trajectory')
        for i in range(len(trajectory) - 1):
            plt.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                        arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->", lw=2))
    elif type == 'scatter':
        plt.grid(False) 
        plt.scatter(x, y, c=colors, cmap='viridis', marker='o', label='Trajectory Points')
    elif type == 'both':
        plt.grid(False) 
        plt.scatter(x, y, c=colors, cmap='viridis', marker='o', label='Trajectory Points')
        plt.plot(x, y, c='gray', linestyle='--', alpha=0.5, label='Path')  
        
    plt.title(f'trajectory') 
    # plt.ylim(80, 200)
    # plt.xlim(190, 300)
    plt.xlabel('x')
    plt.ylabel('y')
    # file_path  = os.path.join(save_dir, f'trajectory/')
    # file_path  = os.path.join(save_dir, f'trajectory/')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'time_{idx}.png'))
    plt.clf()
    plt.close()

def Fourier_filter(x, idx, save_dir=None, fold_name=None):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.type(torch.float32)
    x_freq = fft.fftn(x, dim=(-2, -1)) 
    x_freq = fft.fftshift(x_freq, dim=(-2, -1)) 
    
    dim, T = x_freq.shape
    
    # magnitude_real = torch.real(x_freq)+1
    magnitude_real = torch.abs(x_freq)
    magnitude_min = magnitude_real.min()
    magnitude_max = magnitude_real.max()
    normalized_magnitude = torch.log((magnitude_real - magnitude_min) / (magnitude_max - magnitude_min)+1)
    
    plt.figure()
    plt.imshow(normalized_magnitude.cpu(), cmap='gray', aspect='auto')
    plt.colorbar(label="Normalized Intensity")
    plt.title(f'pusht {dim}x{T} Tensor as Grayscale Image')
    plt.xlabel(f'Columns ({T} horizons)')
    plt.ylabel(f'Rows ({dim} channels)')
    # plt.legend()
    file_path  = os.path.join(save_dir, fold_name)
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(os.path.join(file_path, f'time_{idx}_Fourier_{dim}_{T}.png'))
    plt.clf()
    # plt.savefig('fearture_delta_ddpm.png')
    plt.close()

# pca
def visualize_and_save_features_pca(feature_map, dim, idx,save_dir=None):
    feature_maps = feature_map.cpu().numpy() 
    pca = PCA(n_components=dim) 
    feature_maps_T = feature_maps.T 
    pca.fit(feature_maps_T)
    feature_maps_pca = pca.transform(feature_maps_T)  
    Fourier_filter(feature_maps_pca, idx, save_dir=save_dir, fold_name='Fourier_pca')
    feature_map_norm = (feature_maps_pca - feature_maps_pca.min(axis=0, keepdims=True)) / \
                (feature_maps_pca.max(axis=0, keepdims=True) - feature_maps_pca.min(axis=0, keepdims=True))
    plt.figure()
    # for feature in feature_map_norm.T:
    #     plt.imshow(feature[np.newaxis, :], cmap='gray', aspect='auto')
    plt.imshow(feature_map_norm.T, cmap='gray', aspect='auto')
    plt.title("Feature Grayscale Visualization")
    plt.axis("off")
    file_path  = os.path.join(save_dir, f'feature_pca_grayscale/')
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(os.path.join(file_path, f'time_{idx}.png'))
    plt.close()
    
    feature_visual_block(feature_map_norm, idx, save_dir, type='scatter')
    
def create_gif(image_folder, output_path, duration=0.3):
    images = []
    for file_name in sorted(os.listdir(image_folder)):  
        if file_name.endswith(('png', 'jpg', 'jpeg')):
            file_path = os.path.join(image_folder, file_name)
            images.append(imageio.imread(file_path))
    
    imageio.mimsave(output_path, images, duration=duration)  
    print(f"GIF saved at {output_path}")

def feature_visual(trajectory, output_dir):
    x = trajectory[0][:, 0]
    y = trajectory[0][:, 1]
    # figsize=(int(max(x)-min(x)+3), int(max(y)-min(y)+3))
    

    sigma = 1  
    x_low = gaussian_filter1d(x, sigma=sigma)
    y_low = gaussian_filter1d(y, sigma=sigma)
    
    x_high = x - x_low
    y_high = y - y_low
    
    x_list = [x, x_low, x_high]
    y_list = [y, y_low, y_high]
    name_list = ["Original","low frequency","high frequency component"]
    
    def record_feature(x_list, y_list, output_dir, name_list):
        for tre_x, tre_y, file_name in zip(x_list, y_list, name_list):
            record_path = os.path.join(output_dir, file_name)
            os.makedirs(record_path, exist_ok=True)
            
            plt.figure()
            plt.plot(tre_x, tre_y, marker='o', color='b', linestyle='-', markersize=5)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'{file_name} 2D Trajectory')
            file_path = os.path.join(record_path, f'{file_name}_trajectory.png')
            plt.savefig(file_path)
            # plt.legend()
            plt.grid(True)
            plt.clf()
            plt.close()
        
        return None
    
    record_feature(x_list, y_list, output_dir, name_list)

    
    return None

    
