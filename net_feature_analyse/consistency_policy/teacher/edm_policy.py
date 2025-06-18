from typing import Dict
import math
import os
import pdb
import datetime
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

from consistency_policy.diffusion import Karras_Scheduler, Huber_Loss
from consistency_policy.diffusion_unet_with_dropout import ConditionalUnet1D
from feature_analyse.coefficient import get_key_steps, get_key_steps_edm
from feature_analyse.freq_encoder_utils import register_time, freeu_register_normal_forward
from feature_analyse.feature_utils import feature_visual_block

class KarrasUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: Karras_Scheduler,
            horizon,  
            n_action_steps, 
            n_obs_steps, 
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=128,
            down_dims=(256,512,1024), 
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False, 
            eval_fixed_crop=False, 
            delta=.0,
            inference_mode=False,
):
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
                ac_dim=action_dim, # 10
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
        
        model.prepare_drop_generators() 

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

        self.delta = delta

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
        return mse_loss   
    
    def visual_chart(self, feature_delta):
        cpu_feature_delta = [[tensor.cpu().numpy() for tensor in sublist] for sublist in feature_delta]
        block_name = ["Encoder Block 1","Encoder Block 2","Encoder Block 3","Mid Block 1","Mid Block 2","Decoder Block 1","Decoder Block 2","Final Conv Block"]
        gap_list =  np.arange(1,79)
        plt.figure()
        for i, sub_list in enumerate(cpu_feature_delta):
            if i < 3:  
                plt.plot(gap_list, sub_list, label=block_name[i], linestyle='-')  
            elif i < 5:  
                plt.plot(gap_list, sub_list, label=block_name[i], linestyle='-.')  
            else:
                plt.plot(gap_list, sub_list, label=block_name[i], linestyle='--')
        
        # plt.title('denoisedy_next')
        # plt.xlabel('denoise time step')
        # plt.ylabel('feature_delta')
        # plt.ylabel('noise delta')
        plt.ylim(0, 0.0025)
        # plt.legend()
        plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 78])
        # plt.xticks(np.arange(2,10))

        now = datetime.datetime.now() 
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        taks_name = 'pusht' 
        save_dir = f'./data/{taks_name}_edm_eval_output/feature/' 
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(f'{save_dir}{timestamp}_fearture_delta_edm.png')
        # plt.savefig('fearture_delta_ddpm.png')
        plt.close()
        
        return None
    
    def Fourier_filter(self, x, idx, save_dir=None, fold_name=None):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.type(torch.float32)
        x_freq = fft.fftn(x, dim=(-2, -1)) 
        # x_freq = fft.fftn(x.to('cpu'), dim=(-2, -1)) 
        x_freq = fft.fftshift(x_freq, dim=(-2, -1)) 
        
        dim, T = x_freq.shape
        
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
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            ):
        model = self.model # 1d UNet
        scheduler = self.noise_scheduler 

        trajectory = scheduler.sample_inital_position(condition_data, generator=generator) 
    
        timesteps = torch.arange(0, self.noise_scheduler.bins, device=condition_data.device)
        total_time_feature = []
        total_time_feature_next = []
        energy_list = []
        energy_next_list = []
        for b, next_b in zip(timesteps[:-1], timesteps[1:]):
            trajectory[condition_mask] = condition_data[condition_mask] 

            t = scheduler.timesteps_to_times(b)
            next_t = scheduler.timesteps_to_times(next_b)

            
            denoise = lambda traj, t: model(traj, t, local_cond=local_cond, global_cond=global_cond) 
            # 3. compute previous image: x_t -> x_t-1

            trajectory, feature1, feature2, energy, energy_next = scheduler.step(denoise, trajectory, t, next_t, clamp = True) 

            total_time_feature.append(feature1)
            total_time_feature_next.append(feature2)
            energy_list.append(energy)
            energy_next_list.append(energy_next)
        
        # cc = [math.sqrt(energy_list[idx+1]/energy_list[0]) for idx in range(len(energy_list)-1)]
        # print(f"total energy compensation coefficient{cc}")
        # cc_next = [math.sqrt(energy_next_list[idx+1]/energy_next_list[0]) for idx in range(len(energy_next_list)-1)]
        # print(f"total next step energy compensation coefficient{cc_next}")
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]      

        net_feature = copy.deepcopy(total_time_feature)
        net_featur_next = copy.deepcopy(total_time_feature_next)
        return trajectory, net_feature,  net_featur_next

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict 
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
        if self.obs_as_global_cond: # true
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
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample, net_feature, net_feature_next = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond) 
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]

        start = To - 1 
        end = start + self.n_action_steps 

        action_pred = self.normalizer['action'].unnormalize(naction_pred) 

        # get action
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        
        # ================= feature analyse =================
        feature = {}
        feature_next = {}
        for i in range (len(net_feature[0])): 
            feature[f'block_{i}'] = []
            feature_next[f'block_{i}'] = []
        
        for blocks, blocks_next in zip(net_feature, net_feature_next): 
            for i, block in enumerate(blocks):
                feature[f'block_{i % len(net_feature[0])}'].append(block) 
                # print(len(feature[f'block_{idx % len(net_feature[0])}']))
                # save_dir = os.path.join("./data/freq_edm/", f'block_{i}')
                # os.makedirs(save_dir, exist_ok=True)
                # self.Fourier_filter(block[0], idx, save_dir=save_dir, fold_name='Fourier')
            for j,block in enumerate(blocks_next):
                feature_next[f'block_{j % len(net_feature_next[0])}'].append(block) 
                
        net_feature = []
        for block in feature.values():
            net_feature.append(block) 
        net_feature_next = []
        for block in feature_next.values():
            net_feature_next.append(block) 
 
        Mse = []
        Mse_next = []

        for block, block_next in zip(net_feature, net_feature_next):
            Mse.append(self.mse_caluate(block)) 
            Mse_next.append(self.mse_caluate(block_next))
        # Mse.append(self.mse_caluate(net_feature)) 
        # Mse.append(self.mse_caluate(net_feature_next))

        # feature visualize
        self.visual_chart(Mse)
        # self.visual_chart(Mse_next)
        
        get_key_steps(Mse, key_step_num=3)
        get_key_steps(Mse_next, key_step_num=3) # encoder block2 ,encoder block3, backbone
        
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


        # Sample a random timestep for each image
        times, _ = self.noise_scheduler.sample_times(trajectory) 

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, times) 
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask] 
        
        # Predict the initial state
        denoise = lambda traj, t: self.model(traj, t, local_cond=local_cond, global_cond=global_cond) 
        pred, _ = self.noise_scheduler.calc_out(denoise, noisy_trajectory, times, clamp=False) 
        weights = self.noise_scheduler.get_weights(times, None, "karras") 
        
        target = trajectory 

        loss = Huber_Loss(pred, target, delta = self.delta, weights = weights) # Setting delta = -1 calculates iCT's recommended delta given data size
        return loss


    