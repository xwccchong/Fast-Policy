"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# my_output_dir = "/data/wts/diffusion_policy/data/pusht/outputs/2024.09.29/14.52.41_train_diffusion_unet_hybrid_pusht_image" # 继续训练

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf, output_dir=None):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir)
    workspace.run()

if __name__ == "__main__":
    main()
    
# HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda hydra.run.dir='/data/wts/diffusion_policy/data/pusht/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn_ddim.yaml training.seed=42 training.device=cuda hydra.run.dir='/data/wts/diffusion_policy/data/pusht/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# HYDRA_FULL_ERROR=1 python train.py --config-dir=./diffusion_policy/config/task_cfg/square --config-name=square_mh.yaml training.seed=42 training.device=cuda hydra.run.dir='/data/wts/diffusion_policy/data/square/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# Running for multiple seeds 
# export CUDA_VISIBLE_DEVICES=0,1,2
# ray start --head --num-gpus=3
# (robodiff)[diffusion_policy]$ python ray_train_multirun.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml --seeds=42,43,44 --monitor_key=test/mean_score -- multi_run.run_dir='/data/wts/diffusion_policy/data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}'