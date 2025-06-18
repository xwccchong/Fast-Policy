from typing import Dict
import pdb # pdb.set_trace()
import os
import math
import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import matplotlib.pyplot as plt
import numpy as np
import copy 
import datetime
from matplotlib.cm import viridis
from copy import deepcopy 
from math import sqrt
from PIL import Image
from einops import rearrange, reduce
from sklearn.decomposition import PCA
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from feature_analyse.coefficient import get_key_steps

class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # noise_scheduler: DDIMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        # print(f"input_dim: {input_dim}")
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    def mse_caluate(self, feature):
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
            mse_loss.append(F.mse_loss(feat1, feat2)*1000/(feat1.shape[0]*feat1.shape[1]*feat1.shape[2]))
            # mse_loss.append(F.mse_loss(feat1, feat2)/(feat1.shape[0])) 
        # mse_loss = F.mse_loss(f1, f2)
        
        return mse_loss

    def visual_chart(self, feature_delta):
        cpu_feature_delta = [[tensor.cpu().numpy() for tensor in sublist] for sublist in feature_delta]
        block_name = ["Encoder Block 1","Encoder Block 2","Encoder Block 3","Mid Block 1","Mid Block 2","Decoder Block 1","Decoder Block 2","Final Conv Block"]
        gap_list =  np.arange(1,10)

        plt.figure()
        for i, sub_list in enumerate(cpu_feature_delta):
            # encoder
            if i < 3:  
                plt.plot(gap_list, sub_list, label=block_name[i], linestyle='-')  
            # mid
            elif i < 5:  
                plt.plot(gap_list, sub_list, label=block_name[i], linestyle='-.')  
            # decoder
            else:
                plt.plot(gap_list, sub_list, label=block_name[i], linestyle='--')   

        plt.ylim(0, 0.5)
        # plt.xticks(np.arange(1, 10))  
        # plt.yticks(np.arange(0, 0.5,0.1))
        # plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # plt.legend()
        plt.show()
        
        now = datetime.datetime.now() 
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        taks_name = 'pusht' 
        save_dir = f'./data/{taks_name}_eval_output/feature/' 
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(f'{save_dir}{timestamp}_fearture_delta_DDPM.png')
        # plt.savefig('fearture_delta_ddpm.png')
        plt.close()
        
        return None
    
    def feature_visual_block(self, trajectory, idx, save_dir, type=None):
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
            
        plt.title(f'pac_feature_trajectory') 
        # plt.ylim(0, 0.3)
        plt.xlabel('x')
        plt.ylabel('y')
        file_path  = os.path.join(save_dir, f'pac_trajectory/')
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(os.path.join(file_path, f'time_{idx}.png'))
        plt.clf()
        plt.close()
    
    def Fourier_filter(self, x, idx, save_dir=None, fold_name=None):
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
        h,c = normalized_magnitude.shape
        
        x_min, x_max = int(c/2) - 3, int(c/2) + 4
        y_min, y_max = int(h/2) - 3, int(h/2) + 4
        
        plt.figure()
        # plt.imshow(normalized_magnitude.cpu(), cmap='gray', aspect='auto')
        plt.imshow(normalized_magnitude[int(h/2)-3:int(h/2)+4,int(c/2)-3:int(c/2)+4].cpu(), cmap='gray', aspect='auto') # center
        # plt.xticks(ticks=range(x_min, x_max), labels=range(x_min, x_max))
        # plt.yticks(ticks=range(y_min, y_max), labels=range(y_min, y_max))
        # plt.colorbar(label="Normalized Intensity")
        # plt.title(f'pusht {dim}x{T} Tensor as Spectrum Graph')
        # plt.xlabel(f'channels')
        # plt.ylabel(f'horizons')
        # plt.legend()
        file_path  = os.path.join(save_dir, fold_name)
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(os.path.join(file_path, f'time_{idx}_Fourier_{dim}_{T}.png'))
        plt.clf()
        # plt.savefig('fearture_delta_ddpm.png')
        plt.close()
    
    def visualize_and_save_features_pca(self, feature_map, dim, idx,save_dir=None):
        feature_maps = feature_map.cpu().numpy() 
        pca = PCA(n_components=dim) 
        feature_maps_T = feature_maps.T 
        pca.fit(feature_maps_T)
        feature_maps_pca = pca.transform(feature_maps_T)  
        self. Fourier_filter(feature_maps_pca, idx, save_dir=save_dir, fold_name='Fourier_pca')
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
        
        self.feature_visual_block(feature_map_norm, idx, save_dir, type='scatter')
        
    def create_gif(self, image_folder, output_path, duration=0.3):
        images = []
        for file_name in sorted(os.listdir(image_folder)):  
            if file_name.endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(image_folder, file_name)
                images.append(imageio.imread(file_path))
        
        imageio.mimsave(output_path, images, duration=duration)  
        print(f"GIF saved at {output_path}")
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        
        save_dir = os.path.join("./data/pca/", f'initial_noise')
        os.makedirs(save_dir, exist_ok=True)
        self.feature_visual_block(trajectory[0], idx = 0, save_dir = save_dir,type='scatter')
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        total_time_feature = []
        for t in scheduler.timesteps:  
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output, encoder_feature = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            
            total_time_feature.append(encoder_feature) 
        
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        net_feature = deepcopy(total_time_feature)
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory, net_feature


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample, net_feature = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }

        # Draw trajectory
        self.feature_visual_block(action[0], idx=10, save_dir="./data/pca/", type='both')
        self.feature_visual_block(naction_pred[:,start:end][0], idx=11, save_dir="./data/pca/", type='both')
        
        feature = {}
        for i in range (len(net_feature[0])):
            feature[f'block_{i}'] = []
        
        dim = action[0].shape[-1]
        for idx, blocks in enumerate(net_feature): 
            for i, block in enumerate(blocks):
                feature[f'block_{i % len(net_feature[0])}'].append(block)
                save_dir = os.path.join("./data/pca/", f'block_{i}')
                self.Fourier_filter(block[0], idx, save_dir=save_dir, fold_name='Fourier')
                # if i == len(blocks) - 1:
                #     trej = block[0][:,start:end] # horizon内的action
                #     self.feature_visual_block(trej.T, idx+10, save_dir = save_dir, type='both')
                # self.visualize_and_save_features_pca(block[0], dim, idx, save_dir=save_dir) # 只取第一个batchsize

        # for i in range(8):
        #     save_dir = os.path.join("./data/pca/", f'block_{i}') 
        #     self.create_gif(os.path.join(save_dir, "pac_trajectory"), os.path.join(save_dir, "traj_change.gif"), duration=0.6)
        
        net_feature = []
        for block in feature.values():
            net_feature.append(block)  

        Mse = []
        for block in net_feature:
            Mse.append(self.mse_caluate(block)) 
        self.visual_chart(Mse)
        
        get_key_steps(Mse, key_step_num=3) 

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred, encoder_fearture = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
