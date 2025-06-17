# Fast Policy：Accelerating Visuomotor Policies without Re-training

[[Paper]]()

[Tongshu Wu](https://github.com/xwccchong)<sup>1</sup>,
[Zheng Wang](https://cs.tongji.edu.cn/info/1061/3377.htm)<sup>1</sup>

<sup>1</sup>Tongji University


## Introduction
Fast Policy (termed FP), a **train-free** method, which can be regarded as a powerful and accelerated alternative to Diffusion Policy for learning visuomotor robot control. Specifically, our comprehensive study of UNet encoder shows that its features change little during inference, prompting us to reuse encoder features in non-critical denoising steps. In addition, we design strategies based on Fourier energy to screen critical and non-critical steps dynamically according to different tasks. Importantly, to mitigate performance degradation caused by the repeated use of non-critical steps, we also introduce a noise correction strategy. Our FP is evaluated on multiple simulation benchmarks and the comparison results with existing speed-up methods demonstrate the effectiveness and superiority of FP with state-of-the-art success rates in visuomotor inference speed.

## Installation
Our work is built upon [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [Consistency Policy](https://github.com/Aaditya-Prasad/consistency-policy), so the environment setup follows the installation procedure of Diffusion Policy. Our operating system is Ubuntu 20.04.
Please first clone the project to  local:
```console
$ mamba env create -f conda_environment.yaml
```

Then follow the steps below to set up the environment.
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f conda_environment.yaml
```

but you can use conda as well: 
```console
$ conda env create -f conda_environment.yaml
```

The `conda_environment_macos.yaml` file is only for development on MacOS and does not have full support for benchmarks.

**Note**: My environment was initially set up following the installation instructions of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), and then I additionally installed the required libraries based on the `requirements.txt` from [Faster Diffusion](https://github.com/hutaiHang/Faster-Diffusion). I had provided my `requirements.txt` for reference, so that the dependency libraries can be properly aligned.
```console
pip install -r requirements.txt
```

The pretrained models used in the comparative experiments were trained under the original environments of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [Consistency Policy](https://github.com/Aaditya-Prasad/consistency-policy).

🌟Issues encountered and corresponding solutions during environment setup:
1. While installing the simulation environment for Consistency Policy, I modified the requirements.txt as follows:
```console
- r3m @ https://github.com/facebookresearch/r3m/archive/b2334e726887fa0206962d7984c69c5fb09cceab.tar.gz
- robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz
# r3m==0.0.0
# robosuite==1.2.0
```

2. If the installation process of the simulation environment is interrupted or fails, please reinstall it from scratch. Otherwise, missing library errors may occur during program execution.

3. For setting up the MuJoCo environment, you can refer to:[Mujoco Installation]（https://github.com/Liujian1997/Franka_env-Installation/blob/main/Mujoco安装记录.md）or other websites.

## Trying Fast Policy
```console
cd faster_diffusion_policy
conda activate faster_diffusion_policy
```

First, modify the relevant parameters and dataset paths in the corresponding folders to ensure that the dataset can be loaded correctly.
```console
(faster_diffusion_policy)[./faster_diffusion_policy/diffusion_policy/config/task_cfg/square/square_mh.yaml]dataset_path: <your/dataset/path>
```

Once you have obtained the pretrained model for a specific task, you can extract the key steps using the instructions provided below.
```console
(faster_diffusion_policy)[./faster_diffusion_policy/net_feature_analyse]python eval.py --checkpoint <your/ckpt/path> --output_dir <your/result/output/path> --device cuda:0
```

**Note**: In `eval.py`, I used `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` to restrict the visible GPU. If you are using a single GPU, please do not modify `--device cuda:0`, as the designated GPU index will be 0.If you wish to apply our method to other network architectures, you will need to recalculate the weights based on the energy evaluation results in the code before extracting the key steps again.

After obtaining the key steps corresponding to the pretrained model, you should modify the key steps in `[./faster_diffusion_policy/freq_encoder/freq_encoder_utils.py]` and `.faster_diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`, then start using Fast Policy for accelerated inference!

Then, modify the following function call in the corresponding evaluation script:
```console
(faster_diffusion_policy)[./faster_diffusion_policy/eval.py]ParaCorrection(cfg, step_num=10, dataset_path=None, batch_size=64, scheduler="DDIM", n_envs=None)
```

In the evaluation file list, each evaluation function corresponds to the following settings:
- eval.py → Diffusion Policy
- consistency_eval.py → Consistency Policy without student model
- eval_cp.py → Consistency Policy with student model

Our method only uses `eval.py` and `consistency_eval.py`. You can use Fast Policy for accelerated inference!
```console
(faster_diffusion_policy)[./faster_diffusion_policy]python eval.py --checkpoint <your/DP/ckpt/path> --output_dir <your/result/output/path> --device cuda:0
(faster_diffusion_policy)[./faster_diffusion_policy]python consistency_eval.py --checkpoint <your/CP w/o student model/ckpt/path> --output_dir <your/result/output/path> --device cuda:0
```

## Real experiment
The real experiment will be released together with the project page of the paper soon.

## License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgement
Much of our implementation is adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [Consistency Policy](https://github.com/Aaditya-Prasad/consistency-policy) and [Faster Diffusion](https://github.com/hutaiHang/Faster-Diffusion).
