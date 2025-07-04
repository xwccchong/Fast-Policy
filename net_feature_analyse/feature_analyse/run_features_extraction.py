import pdb # pdb.set_trace()
import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import json
from torchvision import transforms
import logging
from pnp_utils import check_safety

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# export CUDA_VISIBLE_DEVICES=2,3 && python run_features_extraction.py --config configs/pnp/feature-extraction-generated.yaml --save_all_features
# python run_features_extraction.py --config configs/pnp/feature-extraction-real.yaml
# ln -s /data/wts/model_ckpt/sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt

def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS) # 重采样。这是一种高质量的重采样方法，可以在调整图像尺寸时保留更多细节
    image = np.array(image).astype(np.float32) / 255.0 # 归一化
    image = image[None].transpose(0, 3, 1, 2) # 调整维度顺序
    image = torch.from_numpy(image) # 转换为张量
    return 2.*image - 1. # 将像素值从[0, 1]范围缩放到[-1, 1]范围


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}") # 470000
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval() # 设置成评估模式
    return model # 扩散模型


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="configs/pnp/feature-extraction-generated.yaml",
        help="path to the feature extraction config file"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--save_all_features",
        action="store_true", 
        help="if set to true, saves all feature maps, otherwise only saves those necessary for PnP",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--check-safety",
        action='store_true',
    )

    opt = parser.parse_args()
    setup_config = OmegaConf.load("./configs/pnp/setup.yaml")
    model_config = OmegaConf.load(f"{opt.model_config}")
    exp_config = OmegaConf.load(f"{opt.config}")
    exp_path_root = setup_config.config.exp_path_root

    if exp_config.config.init_img != '': # cfg里面不是默认为""?
        exp_config.config.seed = -1
        exp_config.config.prompt = ""
        exp_config.config.scale = 1.0
        
    seed = exp_config.config.seed 
    seed_everything(seed)

    model = load_model_from_config(model_config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model # unet model直接调用了 model.model.diffusion_model，在这个地方要修改为需要测试特征的网络结构
    sampler = DDIMSampler(model) # 采用的ddim采样器
    save_feature_timesteps = exp_config.config.ddim_steps if exp_config.config.init_img == '' else exp_config.config.save_feature_timesteps

    outpath = f"{exp_path_root}/{exp_config.config.experiment_name}" # ./experiments/experiment_name

    callback_timesteps_to_save = [save_feature_timesteps] # 转化为一个list？ [50]
    if os.path.exists(outpath):
        logging.warning("Experiment directory already exists, previously saved content will be overriden")
        if exp_config.config.init_img != '': # 如果不是正常模式，就args
            with open(os.path.join(outpath, "args.json"), "r") as f:
                args = json.load(f) # 将args中的内容读取
            callback_timesteps_to_save = args["save_feature_timesteps"] + callback_timesteps_to_save

    predicted_samples_path = os.path.join(outpath, "predicted_samples")
    feature_maps_path = os.path.join(outpath, "feature_maps")
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(predicted_samples_path, exist_ok=True)
    os.makedirs(feature_maps_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)

    # save parse_args in experiment dir
    # with open(os.path.join(outpath, "args.json"), "w") as f:
    #     args_to_save = OmegaConf.to_container(exp_config.config)
    #     args_to_save["save_feature_timesteps"] = callback_timesteps_to_save
    #     json.dump(args_to_save, f) # 将args中的内容保存到json文件中

    def save_sampled_img(x, i, save_path):
        x_samples_ddim = model.decode_first_stage(x)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0) # 由[-1,1]映射到[0,1]
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy() # 转到cpu上，改变张量的维度顺序，并且转化为numpy？
        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2) # 由np转化为张量并改变张量的维度顺序
        x_sample = x_image_torch[0]
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(save_path, f"{i}.png"))
    
    # 保存采样去噪得到的图像
    def ddim_sampler_callback(pred_x0, xt, i):  # 这个i 没啥用啊？
        save_feature_maps_callback(i) 
        save_sampled_img(pred_x0, i, predicted_samples_path)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        print(f"whether save all features:{opt.save_all_features}")
        for block in tqdm(blocks, desc="Saving input blocks feature maps"):
            if not opt.save_all_features and block_idx < 4:
                block_idx += 1
                continue
            if "ResBlock" in str(type(block[0])):
                if opt.save_all_features or block_idx == 4:
                    save_feature_map(block[0].in_layers_features, f"{feature_type}_{block_idx}_in_layers_features_time_{i}")
                    save_feature_map(block[0].out_layers_features, f"{feature_type}_{block_idx}_out_layers_features_time_{i}") # 所以out_layers_features才是resnet的输出特征，
                    # save_feature_map(block[0].skip_connection_features, f"{feature_type}_{block_idx}_skip_connection_features_time_{i}")
            # if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
            #     save_feature_map(block[1].transformer_blocks[0].attn1.k, f"{feature_type}_{block_idx}_self_attn_k_time_{i}")
            #     save_feature_map(block[1].transformer_blocks[0].attn1.q, f"{feature_type}_{block_idx}_self_attn_q_time_{i}")
            block_idx += 1
        print(block_name_list)

    def save_feature_maps_callback(i):
        if opt.save_all_features:
            save_feature_maps(unet_model.input_blocks, i, "input_block")
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map(feature_map, filename):
        save_path = os.path.join(feature_maps_path, f"{filename}.pt")
        torch.save(feature_map, save_path)

    assert exp_config.config.prompt is not None
    prompts = [exp_config.config.prompt]

    # 记录所有block output模块的name
    block_name_list = []
    blocks = [unet_model.input_blocks, unet_model.output_blocks]
    for item in blocks:
        if item == unet_model.input_blocks:
            block_idx = 0
            for block in tqdm(item, desc="Saving input blocks feature maps"):
                if "ResBlock" in str(type(block[0])):  
                    block_name_list.append(f"input_block_{block_idx}")
                block_idx += 1 
        else:
            block_idx = 0
            for block in tqdm(item, desc="Saving output blocks feature maps"):
                if "ResBlock" in str(type(block[0])):  
                    block_name_list.append(f"output_block_{block_idx}")
                block_idx += 1 
                
    # save parse_args in experiment dir           
    with open(os.path.join(outpath, "args.json"), "w") as f:
        args_to_save = OmegaConf.to_container(exp_config.config)
        args_to_save["save_feature_timesteps"] = callback_timesteps_to_save
        args_to_save["block_name"] = block_name_list
        json.dump(args_to_save, f)  
        
                    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = model.get_learned_conditioning([""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f] # 下采样因子放缩

                z_enc = None
                if exp_config.config.init_img != '': # 如果是真实输入图像，就不是这个了
                    assert os.path.isfile(exp_config.config.init_img)
                    init_image = load_img(exp_config.config.init_img).to(device)
                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
                    ddim_inversion_steps = 999 # 设置去噪共1000步，ddim作用下 进行50次推理
                    z_enc, _ = sampler.encode_ddim(init_latent, num_steps=ddim_inversion_steps, conditioning=c,unconditional_conditioning=uc,unconditional_guidance_scale=exp_config.config.scale)
                else:
                    z_enc = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device) # 随机生成噪声
                torch.save(z_enc, f"{outpath}/z_enc.pt")
                samples_ddim, _ = sampler.sample(S=exp_config.config.ddim_steps,
                                conditioning=c,
                                batch_size=1,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=exp_config.config.scale,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                x_T=z_enc,
                                img_callback=ddim_sampler_callback,
                                callback_ddim_timesteps=save_feature_timesteps,
                                outpath=outpath) # 在这个去噪的每一个时间步中，对特征以及此时的图像进行存储

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                if opt.check_safety:
                    x_samples_ddim = check_safety(x_samples_ddim) # TODO: implement safety check
                x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                sample_idx = 0
                for x_sample in x_image_torch: # 最终去噪的到的图像
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{sample_idx}.png")) # 
                    sample_idx += 1

    print(f"Sampled images and extracted features saved in: {outpath}")


if __name__ == "__main__":
    main()
