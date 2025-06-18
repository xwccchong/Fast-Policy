import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
  
import pdb
import argparse, os
from tqdm import trange
import torch
import torch.nn.functional as F
from einops import rearrange
from pnp_utils import visualize_and_save_features_pca
from omegaconf import OmegaConf
import json
from run_features_extraction import load_model_from_config
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ldm.models.diffusion.ddim import DDIMSampler

# export CUDA_VISIBLE_DEVICES=2,3 && python run_features_pca.py --config configs/pnp/feature-pca-vis.yaml

def load_experiments_features(feature_maps_paths, block, feature_type, t):
    feature_maps = []
    for i, feature_maps_path in enumerate(feature_maps_paths):
        if "attn" in feature_type:
            feature_map = torch.load(os.path.join(feature_maps_path, f"{block}_{feature_type}_time_{t}.pt"))[8:]
            feature_map = rearrange(feature_map, 'h n d -> n (h d)')
        else:
            feature_map = \
                torch.load(os.path.join(feature_maps_path, f"{block}_{feature_type}_time_{t}.pt"))[1]
            feature_map = feature_map.reshape(feature_map.shape[0], -1).t()  # N X C
        feature_maps.append(feature_map)

    # print(f"feature_map number:{len(feature_maps)}")
    # print(f"feature_map shape:{feature_map.shape}")
    
    return feature_maps


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="configs/pnp/feature-pca-vis.yaml",
        help="path to the feature PCA visualization config file"
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

    opt = parser.parse_args()
    setup_config = OmegaConf.load("./configs/pnp/setup.yaml")
    exp_path_root = setup_config.config.exp_path_root
    exp_config = OmegaConf.load(f"{opt.config}")
    transform_experiments = exp_config.config.experiments_transform
    fit_experiments = exp_config.config.experiments_fit

    with open(os.path.join(exp_path_root, transform_experiments[0], "args.json"), "r") as f:
        args = json.load(f)
        ddim_steps = args["save_feature_timesteps"][-1]
        block_name_list = args["block_name"]
        print(f"block_name_list:{block_name_list}")

    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}") # 加载模型

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    total_steps = sampler.ddim_timesteps.shape[0] # 时间步数
    iterator = tqdm(time_range, desc="visualizing features", total=total_steps) # 预定义进度条 时间步

    print(f"visualizing features PCA experiments: block - {exp_config.config.block}; transform experiments - {exp_config.config.experiments_transform}; fit experiments - {exp_config.config.experiments_fit}")

    # 获取特征图存储路径
    transform_feature_maps_paths = []
    for experiment in transform_experiments:
        transform_feature_maps_paths.append(os.path.join(exp_path_root, experiment, "feature_maps"))
    print(f"transform_feature_maps_paths:{transform_feature_maps_paths}")
    
    fit_feature_maps_paths = []
    for experiment in fit_experiments:
        fit_feature_maps_paths.append(os.path.join(exp_path_root, experiment, "feature_maps"))
    print(f"fit_feature_maps_paths:{fit_feature_maps_paths}")
    
    feature_types = [
        "in_layers_features",
        "out_layers_features"
        # "self_attn_q",
        # "self_attn_k"
    ]
    feature_pca_paths = {}

    pca_folder_path = os.path.join(exp_path_root, "PCA_features_vis", exp_config.config.experiment_name) # 定义存储路径
    os.makedirs(pca_folder_path, exist_ok=True)

    for feature_type in feature_types:
        feature_pca_path = os.path.join(pca_folder_path, f"{exp_config.config.block}_{feature_type}")
        feature_pca_paths[feature_type] = feature_pca_path
        os.makedirs(feature_pca_path, exist_ok=True)

    def mse_caluate(feature, feature_type):
        if isinstance(feature, list):
            # 确保 feature 至少有两个元素
            if len(feature) < 2:
                raise ValueError("Feature list must contain at least two elements.")
            # feature_cpu = []
            # for item in feature:
            #     # print(type(item))
            #     if torch.is_tensor(item):  # 如果元素是张量
            #         # 将 GPU 张量移动到 CPU
            #         # print(item.shape)
            #         feature_cpu.append(item.detach().cpu().numpy())
            #     else:
            #         # 如果不是张量，直接添加
            #         feature_cpu.append(item) 
            
            # # 将处理后的列表转换为 NumPy 数组
            # feature = np.array(feature_cpu, dtype=np.float32)
            # # 构造前后交错数组 
            # f1 = torch.tensor(feature_cpu[1:], dtype=torch.float32).to(device)
            # f2 = torch.tensor(feature_cpu[:-1], dtype=torch.float32).to(device)
            f1 = feature[1:] # list中包含多个tensor
            f2 = feature[:-1]
        else:
            raise ValueError("feature should be a list")

        assert len(f1) == len(f2)
        # assert f1.shape == f2.shape
        
        mse_loss = []
        for (feat1, feat2) in zip(f1, f2):
            mse_loss.append(F.mse_loss(feat1, feat2))
        # mse_loss = F.mse_loss(f1, f2)
        # print(mse_loss)
        # print(f"{feature_type} MSE Loss:{mse_loss.item()}")
        return mse_loss

    def visual_chart(feature_delta, block_name_list):
        # tensor转移到cpu上
        cpu_feature_delta = [[tensor.cpu().numpy() for tensor in sublist] for sublist in feature_delta]
        
        plt.figure()
        for i, sub_list in enumerate(cpu_feature_delta):
            plt.plot(sub_list, label=block_name_list[i])

        plt.xlabel('denoise time step')
        plt.ylabel('feature_delta')
        plt.legend()
        plt.savefig('fearture delta.png')
        return None
    
    total_in_features = []
    total_out_features = []
    
    # for t in iterator: # t=981 961 ....
    #     for feature_type in feature_types:
    #         fit_features = load_experiments_features(fit_feature_maps_paths, exp_config.config.block, feature_type, t)  # N X C
    #         transform_features = load_experiments_features(transform_feature_maps_paths, exp_config.config.block, feature_type, t)
    #         # 在此时得到两个feature
    #         # visualize_and_save_features_pca(torch.cat(fit_features, dim=0),
    #         #                                 torch.cat(transform_features, dim=0),
    #         #                                 transform_experiments,
    #         #                                 t,
    #         #                                 feature_pca_paths[feature_type])
    #         # print(t)
    #         if feature_type=="in_layers_features":
    #             total_in_features.append(fit_features[0])
    #         elif feature_type=="out_layers_features":
    #             total_out_features.append(fit_features[0])
    
    # print(f"total_in_features num:{len(total_in_features)}")
    # print(f"total_out_features num:{len(total_out_features)}")

    #现在要对比不同block在同一时间步的特征结果
    for feature_type in feature_types: 
        for block_name in block_name_list:
            # 逐个 block 获取特征
            block_fit_features = []
            block_transform_features = []
            for t in iterator:  # t=981, 961, ...
                # 获取当前 block 在 t 时间步的特征
                fit_features = load_experiments_features(fit_feature_maps_paths, block_name, feature_type, t)  # N X C
                transform_features = load_experiments_features(transform_feature_maps_paths, block_name, feature_type, t)
                
                block_fit_features.append(fit_features[0]) # 有多个实验属性，只取第一个
                block_transform_features.append(transform_features[0])
                
                # 此时记录下当前t下的所有层特征 为一个list   
                # 将 fit_features 和 transform_features 进行 PCA 或其他处理
                # visualize_and_save_features_pca(fit_cat_features, transform_cat_features, transform_experiments, t, feature_pca_paths[feature_type])

                # 根据不同 feature_type 进行存储
            if feature_type == "in_layers_features":
                total_in_features.append(block_fit_features) 
                # total_in_features.append(block_transform_features)
            elif feature_type == "out_layers_features":
                total_out_features.append(block_fit_features)
                # total_out_features.append(block_transform_features)
                    
            
    print(f"total_in_features step num:{len(total_in_features)}") # 20
    print(f"total_out_features step num:{len(total_out_features)}")
    print(f"total_in_features num:{len(total_in_features[0])}") # 50
    print(f"total_out_features num:{len(total_out_features[0])}")

    # 得到时间步之间的mse
    mse_in = []
    mse_out = []
    # pdb.set_trace()
    for f_in in total_in_features:
        mse_in.append(mse_caluate(f_in,feature_type = "in_layers_features"))
    for f_out in total_out_features:
        mse_out.append(mse_caluate(f_out,feature_type = "out_layers_features"))
    # print(f"mse_in:{mse_in}")
    # print(f"mse_out:{mse_out}")
    
    visual_chart(mse_out, block_name_list)
    
        

if __name__ == "__main__":
    main()
