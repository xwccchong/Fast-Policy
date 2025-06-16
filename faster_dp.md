# 1 UNet特征分析
文件路径：`[net_feature_analyse/diffusion_policy]./run_features_extraction.py`; 之后运行一遍dp就可以得到

# 2 evalute
由于checkpoints中包含了原先的虚拟环境模型文件等，所以此时无法增加新的env_runner文件，所以此时对env_runner文件进行修改，将原先的普通轨迹预测换成现在加速后的轨迹预测函数`policy.faster_predict_action`

每一个任务都有对应的workspace和policy，要去对应的policy class中添加`faster_predict_action`函数，才能在评估的时候直接调用

其中`faster_conditional_sample`不需要修改，直接拷贝即可；`faster_predition_action`函数则拷贝原先dp的代码之后，将采样函数名称修改为`faster_conditional_sample`即可。同时由于时间步模式的选取，policy类的预定义中需要添加mod参数来确定类型，同时将mod定义为self.mod用于初始化

    class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
        def __init__(self, 
                model: ConditionalUnet1D,
                noise_scheduler: DDPMScheduler,
                horizon, 
                obs_dim, 
                action_dim, 
                n_action_steps, 
                n_obs_steps,
                num_inference_steps=None,
                obs_as_local_cond=False,
                obs_as_global_cond=False,
                pred_action_steps_only=False,
                oa_step_convention=False,
                mod = 'ununi'   # 添加这个参数
                # parameters passed to step
                **kwargs):

# 3 eval指令

# 4 将dp采样器设置为ddim

    noise_scheduler:
        _target_: diffusers.schedulers.scheduling_ddim.DDIMScheduler
        beta_end: 0.02
        beta_schedule: squaredcos_cap_v2
        beta_start: 0.0001
        clip_sample: true
        num_train_timesteps: 100
        prediction_type: epsilon
        set_alpha_to_one: true
        steps_offset: 0
        # rescale_betas_zero_snr: False
        # variance_type: fixed_small
    num_inference_steps: 10 # 100

终端指令修改为：

    # HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn_ddim.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='/data/wts/diffusion_policy/data/pusht/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

库调用修改：在对应的workspace对应的policy前面调用DDIM库

    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# 5 由于数据集结构更改，修改hydra的路径

    hydra.run.dir='/data/wts/diffusion_policy/data/pusht/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
    hydra.run.dir='/data/wts/diffusion_policy/data/<task name>/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# 6 eval ddpm->ddim

    # ddpm schedule -> ddim schedule
    cfg.policy.noise_scheduler = {
    '_target_': 'diffusers.schedulers.scheduling_ddim.DDIMScheduler',
    'beta_end': 0.02,
    'beta_schedule': 'squaredcos_cap_v2',
    'beta_start': 0.0001,
    'clip_sample': True,
    'num_train_timesteps': 100,
    'prediction_type': 'epsilon',
    'set_alpha_to_one': True,
    'steps_offset': 0,
    # 'rescale_betas_zero_snr': False
    # 'variance_type': 'fixed_small'
    }
    cfg.policy.num_inference_steps = 20 

    policy的初始定义中时间步设置属性修改一下

# 7 使用ddim采样特征变化
使用ddim采样特征变化与ddpm相反！这是个很大的问题

# 8 faster方法代码问题
训练出来的模型和基础的dp还有deepcache不一样，代码修改影响了原先的模型结构，肯定是有问题的。

# 9 各个任务ckpt下载
## can，其余同理

    wget -O epoch=1150-test_mean_score=1.000.ckpt https://diffusion-policy.cs.columbia.edu/data/experiments/image/can_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=1150-test_mean_score=1.000.ckpt

# 10 各个任务模型训练

- 修改数据集路径到对应任务
- 修改训练epoch=1000
- 修改wandb启动项，修改名称
- 修改设备label
- 修改task_name

# 11 数据集缓存区存放

    if use_cache:
            cache_zarr_path = dataset_path + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer)


# 程序进程号
tansport_mh [2] 97094
tansport_ph [1] 1915074
tool_hang_ph [1] 128262
pusht [1] 2739819

并行pusht [1] 3674636
并行 43，44 transport_mh [1] 39614
并行 43，44 tool_hang_ph [1] 3592172
并行 43，44 square_mh [2] 1125123
并行 43，44 transport_mh [1] 1566150
并行 43，44 square_ph 
ray start --address='100.84.207.242:6379'


# freeu method
只用在评估上面，即插即用，训练时不使用

    # --------------- FreeU code -----------------------
    # Only operate on the first two stages
    if hidden_states.shape[1] == 1280:
        hidden_states[:,:640] = hidden_states[:,:640] * self.b1 # 为什么不是根据结构来计算的？
        res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=self.s1) # 仅仅是对特征输入进行修改了
    if hidden_states.shape[1] == 640:
        hidden_states[:,:320] = hidden_states[:,:320] * self.b2
        res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=self.s2)
    # ---------------------------------------------------------


# 对加速前后每一个时间步的噪声差异进行可视化
在utils里面修改的，正常的时候可以不加，来提高代码运行速度

    [conditional_sample]
    noise_origin = []   
    noise_origin.append(model_output)
    return trajectory, noise_origin
    
    [predict action]
    nsample, noise_origin = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
    return result, noise_origin

    [faster_conditional_sample]
    noise_faster = []
    for idx, timestep in enumerate(time_ls):
        # print(f"timestep:{timestep}")
        # if timestep/100 < 0.5: # 去噪后半段 50时间步 总共是100步
        #     trajectory = trajectory + 0.003*pre_trajectory # 噪声预注入 加载初始噪声上，而不是预测噪声
        curr_noise = model_output[idx*bs_perstep:(idx+1)*bs_perstep]
        noise_faster.append(curr_noise)

    [faster_conditional_sample]
    nsample, noise_faster = self.faster_conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

    action_dict, noise_origin = policy.predict_action(obs_dict) # 使用加速的来进行评估
    action_dict, noise_faster = policy.faster_predict_action(obs_dict)
    noise_mse = []
    for i in range(len(noise_origin)):
        noise_mse.append((torch.nn.functional.mse_loss(noise_faster[i], noise_origin[i])).cpu().numpy())

    # pdb.set_trace()
    plt.figure()
    plt.grid(True)
    # 创建直方图 
    plt.plot(noise_mse, marker='o', markersize=8, linewidth=2, markerfacecolor='red') 
    # # 添加标题和标签 
    plt.title('noise_delta between time steps') 
    # plt.ylim(0, 0.3)
    plt.xlabel('denoise time step')
    plt.ylabel('noise_delta')
    # plt.legend()
    filename = os.path.join('./data', f'noise_delta_ddim_10_{chunk_idx}.png')
    plt.savefig(filename)
    plt.clf()
    # plt.savefig('fearture_delta_ddpm.png')
    plt.close()

# 问题
当同时运行原始dp和加速dp时，理论上结果不是应该被加速dp覆盖吗？并不会影响到最终仿真结果啊，但是结果却变化了

    action_dict, noise_origin = policy.predict_action(obs_dict) # 使用加速的来进行评估
    action_dict, noise_faster = policy.faster_predict_action(obs_dict)

# freeu setattr
为 FreeU 调整提供自定义参数： b1, b2, s1 和 s2 是在 UNet 网络特定阶段上对特征图进行调整的参数。具体来说：

b1 和 b2 用于缩放 hidden_states 的特定部分，影响特征的权重。
s1 和 s2 用于 Fourier_filter 函数的缩放（scale）操作，这里是对 res_hidden_states 的特征进行修改，可能用于抑制高频噪声或增强特定频率的特征。
在 up_forward 中动态使用这些参数： 这些参数被动态地传递给 up_forward 函数内的特征调整操作，这样每次调用 UpBlock2D 的 forward 方法时，它们都会直接应用。这种设计允许不同的 UpBlock2D 使用不同的参数值，提升了模块化和灵活性。

循环中的 setattr 是为了批量赋值： 最后的循环中使用 setattr 的目的是批量赋值，确保每个符合条件的 UpBlock2D 都能够使用这些参数，而不需要手动为每个块单独设置。这使得代码更简洁，也方便后续修改 b1, b2, s1, s2 的值来调整模型行为。

    for i, upsample_block in enumerate(model.unet.up_blocks):
        if isinstance_str(upsample_block, "UpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)
            setattr(upsample_block, 'b1', b1)
            setattr(upsample_block, 'b2', b2)
            setattr(upsample_block, 's1', s1)
            setattr(upsample_block, 's2', s2)

# RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR
对tensor进行计算：

    p fft.fftn(x[0], dim=(-2, -1))
    *** RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR

先转移到cpu上：

    p fft.fftn(x[0].to('cpu'), dim=(-2, -1))
    tensor([[-433.0321+0.0000j,   47.0326-52.6668j,   -1.2581+0.0000j,
            47.0326+52.6668j],
            [  67.5678-62.7656j,    1.7189+65.4459j,   -1.0392+36.0898j,
            -5.7320+40.0650j],
            [  67.5906+110.8781j,   11.8933-16.7198j,   -4.6534+12.2361j,
            -17.5528+21.5753j],
            ...,
            [ -39.5894+50.2748j,  -21.0544-18.9254j,   17.0074+7.4239j,
            47.8957-22.9126j],
            [  67.5906-110.8781j,  -17.5528-21.5753j,   -4.6534-12.2361j,
            11.8933+16.7198j],
            [  67.5678+62.7656j,   -5.7320-40.0650j,   -1.0392-36.0898j,
                1.7189-65.4459j]])


# torch适配cuda版本库安装

    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

##

    >>> from numba.np.ufunc import _internal
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/home/wts/.conda/envs/faster_diffusion_policy/lib/python3.9/site-packages/numba/np/ufunc/__init__.py", line 3, in <module>
        from numba.np.ufunc.decorators import Vectorize, GUVectorize, vectorize, guvectorize
    File "/home/wts/.conda/envs/faster_diffusion_policy/lib/python3.9/site-packages/numba/np/ufunc/decorators.py", line 3, in <module>
        from numba.np.ufunc import _internal
    SystemError: initialization of _internal failed without raising an exception

thank you for reporting this. Currently, no released version of Numba supports NumPy 1.25
We will close this now. Next release of Numba 0.58 will support 1.25

## eval 指令

    rm -r ./data/fft_fig &&
    # python eval.py --checkpoint /data/wts/diffusion_policy/data/pusht/outputs/2024.11.22/15.52.46_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt --output_dir data/pusht_eval_output --device cuda:0

    # python eval.py --checkpoint /data/wts/diffusion_policy/data/square/outputs/2024.11.08/17.08.14_train_diffusion_unet_hybrid_square_image_ph/checkpoints/latest.ckpt --output_dir data/square_ph_eval_output --device cuda:0

    # python eval.py --checkpoint /data/wts/diffusion_policy/data/square/outputs/2024.11.08/17.10.40_train_diffusion_unet_hybrid_square_image_mh/checkpoints/latest.ckpt --output_dir data/square_mh_eval_output --device cuda:0

    # python eval.py --checkpoint /data/wts/diffusion_policy/data/transport/outputs/2024.11.11/20.10.05_train_diffusion_unet_hybrid_transport_image_ph/checkpoints/latest.ckpt --output_dir data/transport_ph_eval_output --device cuda:0

    # python eval.py --checkpoint /data/wts/diffusion_policy/data/transport/outputs/2024.11.19/08.33.41_train_diffusion_unet_hybrid_transport_image_mh/checkpoints/epoch=1000-test_mean_score=0.660.ckpt --output_dir data/transport_mh_eval_output --device cuda:0

    # python eval.py --checkpoint /data/wts/diffusion_policy/data/tool_hang/outputs/2024.11.19/09.42.38_train_diffusion_unet_hybrid_tool_hang_image_abs/checkpoints/epoch=1000-test_mean_score=0.900.ckpt --output_dir data/tool_hang_eval_output --device cuda:0

    # rm -r ./data/fft_fig && rm -r ./data/pusht_eval_output && python eval.py --checkpoint /data/wts/diffusion_policy/data/pusht/outputs/2024.11.22/15.52.46_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt --output_dir data/pusht_eval_output --device cuda:0
    
    # pusht
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.11.23/14.27.07_train_diffusion_unet_hybrid_pusht_image/train_0/checkpoints/latest.ckpt --output_dir data/pusht/pusht_eval_output_0 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.11.23/14.27.07_train_diffusion_unet_hybrid_pusht_image/train_1/checkpoints/latest.ckpt --output_dir data/pusht/pusht_eval_output_1 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.11.23/14.27.07_train_diffusion_unet_hybrid_pusht_image/train_2/checkpoints/latest.ckpt --output_dir data/pusht/pusht_eval_output_2 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.11.23/14.27.07_train_diffusion_unet_hybrid_pusht_image/train_0/checkpoints/epoch=0350-test_mean_score=0.871.ckpt --output_dir data/pusht/pusht_eval_output_3 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.11.23/14.27.07_train_diffusion_unet_hybrid_pusht_image/train_1/checkpoints/epoch=0600-test_mean_score=0.883.ckpt --output_dir data/pusht/pusht_eval_output_4 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.11.23/14.27.07_train_diffusion_unet_hybrid_pusht_image/train_2/checkpoints/epoch=0650-test_mean_score=0.913.ckpt --output_dir data/pusht/pusht_eval_output_5 --device cuda:0
    
    # square_mh 
    python eval.py --checkpoint /data/wts/diffusion_policy/data/square/outputs/2024.11.08/17.10.40_train_diffusion_unet_hybrid_square_image_mh/checkpoints/latest.ckpt --output_dir data/square_mh/square_mh_eval_output_0 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/square_mh/train_0/2024.12.29/22.58.36_train_diffusion_unet_hybrid_square_image_mh/checkpoints/latest.ckpt --output_dir data/square_mh/square_mh_eval_output_1 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/square_mh/train_1/2024.12.29/23.01.51_train_diffusion_unet_hybrid_square_image_mh/checkpoints/latest.ckpt --output_dir data/square_mh/square_mh_eval_output_2 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/square/outputs/2024.11.08/17.10.40_train_diffusion_unet_hybrid_square_image_mh/checkpoints/epoch=0750-test_mean_score=0.920.ckpt --output_dir data/square_mh/square_mh_eval_output_3 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/square_mh/train_0/2024.12.29/22.58.36_train_diffusion_unet_hybrid_square_image_mh/checkpoints/epoch=0750-test_mean_score=0.880.ckpt --output_dir data/square_mh/square_mh_eval_output_4 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/square_mh/train_1/2024.12.29/23.01.51_train_diffusion_unet_hybrid_square_image_mh/checkpoints/epoch=0750-test_mean_score=0.920.ckpt --output_dir data/square_mh/square_mh_eval_output_5 --device cuda:0

    # square_ph
    python eval.py --checkpoint /data/wts/diffusion_policy/data/square/outputs/2024.11.08/17.08.14_train_diffusion_unet_hybrid_square_image_ph/checkpoints/latest.ckpt --output_dir data/square_ph/square_ph_eval_output_0 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/square_ph/21.51.39_train_diffusion_unet_hybrid_square_image_ph/checkpoints/latest.ckpt --output_dir data/square_ph/square_ph_eval_output_1 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/square_ph/22.26.08_train_diffusion_unet_hybrid_square_image_ph/checkpoints/latest.ckpt --output_dir data/square_ph/square_ph_eval_output_2 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/square/outputs/2024.11.08/17.08.14_train_diffusion_unet_hybrid_square_image_ph/checkpoints/epoch=0900-test_mean_score=0.980.ckpt --output_dir data/square_ph/square_ph_eval_output_3 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/square_ph/21.51.39_train_diffusion_unet_hybrid_square_image_ph/checkpoints/epoch=0900-test_mean_score=0.940.ckpt --output_dir data/square_ph/square_ph_eval_output_4 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/square_ph/22.26.08_train_diffusion_unet_hybrid_square_image_ph/checkpoints/epoch=0900-test_mean_score=0.940.ckpt --output_dir data/square_phsquare_ph_eval_output_5 --device cuda:0

    # transport_ph
    python eval.py --checkpoint /data/wts/diffusion_policy/data/transport/outputs/2024.11.11/20.10.05_train_diffusion_unet_hybrid_transport_image_ph/checkpoints/latest.ckpt --output_dir data/transport_ph/transport_ph_eval_output_0 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.12.14/16.23.35_train_diffusion_unet_hybrid_transport_image_ph/train_0/checkpoints/latest.ckpt --output_dir data/transport_ph/transport_ph_eval_output_1 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.12.14/16.23.35_train_diffusion_unet_hybrid_transport_image_ph/train_1/checkpoints/latest.ckpt --output_dir data/transport_ph/transport_ph_eval_output_2 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/transport/outputs/2024.11.11/20.10.05_train_diffusion_unet_hybrid_transport_image_ph/checkpoints/epoch=0950-test_mean_score=0.960.ckpt --output_dir data/transport_ph/transport_ph_eval_output_3 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.12.14/16.23.35_train_diffusion_unet_hybrid_transport_image_ph/train_0/checkpoints/epoch=0550-test_mean_score=0.920.ckpt --output_dir data/transport_ph/transport_ph_eval_output_4 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.12.14/16.23.35_train_diffusion_unet_hybrid_transport_image_ph/train_1/checkpoints/epoch=0550-test_mean_score=0.960.ckpt --output_dir data/transport_ph/transport_ph_eval_output_5 --device cuda:0

    # transport_mh
    python eval.py --checkpoint /data/wts/diffusion_policy/data/transport/outputs/2024.11.19/08.33.41_train_diffusion_unet_hybrid_transport_image_mh/checkpoints/epoch=1000-test_mean_score=0.660.ckpt --output_dir data/transport_mh/transport_mh_eval_output_0 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.11.30/19.05.59_train_diffusion_unet_hybrid_transport_image_mh/train_0/checkpoints/latest.ckpt --output_dir data/transport_mh/transport_mh_eval_output_1 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.11.30/19.05.59_train_diffusion_unet_hybrid_transport_image_mh/train_1/checkpoints/latest.ckpt --output_dir data/transport_mh/transport_mh_eval_output_2 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/transport/outputs/2024.11.19/08.33.41_train_diffusion_unet_hybrid_transport_image_mh/checkpoints/epoch=0900-test_mean_score=0.760.ckpt --output_dir data/transport_mh/transport_mh_eval_output_3 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.11.30/19.05.59_train_diffusion_unet_hybrid_transport_image_mh/train_0/checkpoints/epoch=0750-test_mean_score=0.740.ckpt --output_dir data/transport_mh/transport_mh_eval_output_4 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.11.30/19.05.59_train_diffusion_unet_hybrid_transport_image_mh/train_1/checkpoints/epoch=0700-test_mean_score=0.720.ckpt --output_dir data/transport_mh/transport_mh_eval_output_5 --device cuda:0
    
    # tool_hang
    python eval.py --checkpoint /data/wts/diffusion_policy/data/tool_hang/outputs/2024.11.19/09.42.38_train_diffusion_unet_hybrid_tool_hang_image_abs/checkpoints/epoch=1000-test_mean_score=0.900.ckpt --output_dir data/tool_hang/tool_hang_eval_output_0 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.12.19/19.13.13_train_diffusion_unet_hybrid_tool_hang_image_abs/train_0/checkpoints/latest.ckpt --output_dir data/tool_hang/tool_hang_eval_output_1 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.12.19/19.13.13_train_diffusion_unet_hybrid_tool_hang_image_abs/train_1/checkpoints/latest.ckpt --output_dir data/tool_hang/tool_hang_eval_output_2 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/tool_hang/outputs/2024.11.19/09.42.38_train_diffusion_unet_hybrid_tool_hang_image_abs/checkpoints/epoch=0700-test_mean_score=0.980.ckpt --output_dir data/tool_hang/tool_hang_eval_output_3 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.12.19/19.13.13_train_diffusion_unet_hybrid_tool_hang_image_abs/train_0/checkpoints/epoch=0400-test_mean_score=0.960.ckpt --output_dir data/tool_hang/tool_hang_eval_output_4 --device cuda:0
    python eval.py --checkpoint /data/wts/diffusion_policy/data/outputs/2024.12.19/19.13.13_train_diffusion_unet_hybrid_tool_hang_image_abs/train_1/checkpoints/epoch=0600-test_mean_score=0.880.ckpt --output_dir data/tool_hang/tool_hang_eval_output_5 --device cuda:0



## 多gpu集群训练

    Local node IP: 100.84.65.111

    --------------------
    Ray runtime started.
    --------------------

    Next steps
    To add another node to this Ray cluster, run
        ray start --address='100.84.65.111:6379'
    
    To connect to this Ray cluster:
        import ray
        ray.init()
    
    To submit a Ray job using the Ray Jobs CLI:
        RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python my_script.py
    
    See https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html 
    for more information on submitting Ray jobs to the Ray cluster.
    
    To terminate the Ray runtime, run
        ray stop
    
    To view the status of the cluster, use
        ray status
    
    To monitor and debug Ray, view the dashboard at 
        127.0.0.1:8265
### 报错
报错：File "/home/wutongshu/.conda/envs/robodiff/lib/python3.9/site-packages/ray/util/serialization_addons.py", line 21, in register_pydantic_serializer
    pydantic.fields.ModelField,
AttributeError: module 'pydantic.fields' has no attribute 'ModelField'

这个错误提示 pydantic.fields 模块中没有属性 ModelField，通常是由于 pydantic 版本不兼容造成的。Ray 可能要求较早版本的 pydantic，而当前环境中安装的是较新版本，其中 ModelField 类的位置或定义已经发生变化

    pip install -U ray

## consistency-policy
论文中所有任务均为ph，教师模型的epoch是400 ，所以先训练400个epoch，如果要对齐onedp，就之后再训练

在测试读取模型的时候，cp的cfg._target_会将workspace索引到cp文件夹下的baseworkspace，所以将这个baseworkspace替换为dp的baseworkspace，然后用dp的eval文件来测试。

cp的网络训练用的data平行并行的跑法，所以模型的key有前缀module，而dp的模型key没有前缀，所以需要将前缀去掉。再baseworkspace中修改一下代码

测试时，运行`consistency_eval.py`

tool_hang 1712615  [1] 3995277

    # HYDRA_FULL_ERROR=1 python train.py --config-dir=./consistency_policy/configs/ --config-name=edm_square.yaml logging.name=edm_square
    # nohup python train.py --config-dir=configs/ --config-name=edm_tool_hang.yaml logging.name=edm_tool_hang > /home/wts/mywork/train_log/edm_tool_hang_ph_output.log 2>&1 &
    # eval
    # square_ph done
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/square/checkpoints/latest.ckpt --output_dir data/square_ph/square_edm_eval_output_0 --device cuda:0 
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/square/checkpoints/epoch=0050-test_mean_score=0.940.ckpt --output_dir data/square_ph/square_edm_eval_output_1 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/square/checkpoints/epoch=0100-test_mean_score=0.900.ckpt --output_dir data/square_ph/square_edm_eval_output_2 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/square/checkpoints/epoch=0150-test_mean_score=0.900.ckpt --output_dir data/square_ph/square_edm_eval_output_3 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/square/checkpoints/epoch=0200-test_mean_score=0.920.ckpt --output_dir data/square_ph/square_edm_eval_output_4 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/square/checkpoints/epoch=0300-test_mean_score=0.840.ckpt --output_dir data/square_ph/square_edm_eval_output_5 --device cuda:0

    # can_ph done
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/can/checkpoints/latest.ckpt --output_dir data/can/can_edm_eval_output_0 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/can/checkpoints/epoch=0350-test_mean_score=0.980.ckpt --output_dir data/can/can_edm_eval_output_1 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/can/checkpoints/epoch=0500-test_mean_score=0.980.ckpt --output_dir data/can/can_edm_eval_output_2 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/can/checkpoints/epoch=0750-test_mean_score=0.980.ckpt --output_dir data/can/can_edm_eval_output_3 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/can/checkpoints/epoch=0800-test_mean_score=0.980.ckpt --output_dir data/can/can_edm_eval_output_4 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/can/checkpoints/epoch=0850-test_mean_score=0.980.ckpt --output_dir data/can/can_edm_eval_output_5 --device cuda:0

    # lift_ph 
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/lift/checkpoints/latest.ckpt --output_dir data/lift/lift_edm_eval_output_0 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/lift/checkpoints/epoch=0050-test_mean_score=1.000.ckpt --output_dir data/lift/lift_edm_eval_output_1 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/lift/checkpoints/epoch=0100-test_mean_score=1.000.ckpt --output_dir data/lift/lift_edm_eval_output_2 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/lift/checkpoints/epoch=0150-test_mean_score=1.000.ckpt --output_dir data/lift/lift_edm_eval_output_3 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/lift/checkpoints/epoch=0200-test_mean_score=1.000.ckpt --output_dir data/lift/lift_edm_eval_output_4 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/lift/checkpoints/epoch=0250-test_mean_score=1.000.ckpt --output_dir data/lift/lift_edm_eval_output_5 --device cuda:0

    # transport_ph done
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/transport/latest.ckpt --output_dir data/tansport_ph/transport_edm_eval_output_0 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/transport/epoch=0050-test_mean_score=0.880.ckpt --output_dir data/tansport_ph/transport_edm_eval_output_1 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/transport/epoch=0300-test_mean_score=0.880.ckpt --output_dir data/tansport_ph/transport_edm_eval_output_2 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/transport/epoch=0500-test_mean_score=0.880.ckpt --output_dir data/tansport_ph/transport_edm_eval_output_3 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/transport/epoch=0800-test_mean_score=0.860.ckpt --output_dir data/tansport_ph/transport_edm_eval_output_4 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/1000epoch/transport/epoch=0950-test_mean_score=0.860.ckpt --output_dir data/tansport_ph/transport_edm_eval_output_5 --device cuda:0

    # pusht
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/pusht/checkpoints/epoch=1000-test_mean_score=0.846.ckpt --output_dir data/pusht/pusht_edm_eval_output_0 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/pusht/checkpoints/epoch=0150-test_mean_score=0.851.ckpt --output_dir data/pusht/pusht_edm_eval_output_1 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/pusht/checkpoints/epoch=0300-test_mean_score=0.837.ckpt --output_dir data/pusht/pusht_edm_eval_output_2 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/pusht/checkpoints/epoch=0600-test_mean_score=0.843.ckpt --output_dir data/pusht/pusht_edm_eval_output_3 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/pusht/checkpoints/epoch=0650-test_mean_score=0.853.ckpt --output_dir data/pusht/pusht_edm_eval_output_4 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/pusht/checkpoints/epoch=0750-test_mean_score=0.854.ckpt --output_dir data/pusht/pusht_edm_eval_output_5 --device cuda:0

    # tool_hang
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/tool_hang_cp/checkpoints/latest.ckpt --output_dir data/tool_hang/tool_hang_edm_eval_output_0 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/tool_hang_cp/checkpoints/epoch=0200-test_mean_score=0.667.ckpt --output_dir data/tool_hang/tool_hang_edm_eval_output_1 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/tool_hang_cp/checkpoints/epoch=0350-test_mean_score=0.667.ckpt --output_dir data/tool_hang/tool_hang_edm_eval_output_2 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/tool_hang_cp/checkpoints/epoch=0750-test_mean_score=0.633.ckpt --output_dir data/tool_hang/tool_hang_edm_eval_output_3 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/tool_hang_cp/checkpoints/epoch=0800-test_mean_score=0.800.ckpt --output_dir data/tool_hang/tool_hang_edm_eval_output_4 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/tool_hang_cp/checkpoints/epoch=0950-test_mean_score=0.800.ckpt --output_dir data/tool_hang/tool_hang_edm_eval_output_5 --device cuda:0

    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/tool_hang_cp/checkpoints/epoch=0500-test_mean_score=0.600.ckpt --output_dir data/tool_hang/tool_hang_edm_eval_output_6 --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/tool_hang_cp/checkpoints/epoch=0300-test_mean_score=0.633.ckpt --output_dir data/tool_hang/tool_hang_edm_eval_output_7 --device cuda:0

    rm -r ./data/fft_fig &&

## cp eval
### pusht
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/pusht/checkpoints/epoch=0450-test_mean_score=0.747.ckpt --output_dir data/cp/pusht_0 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/pusht/checkpoints/epoch=0200-test_mean_score=0.741.ckpt --output_dir data/cp/pusht_1 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/pusht/checkpoints/epoch=0250-test_mean_score=0.748.ckpt --output_dir data/cp/pusht_2 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/pusht/checkpoints/epoch=0350-test_mean_score=0.757.ckpt --output_dir data/cp/pusht_3 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/pusht/checkpoints/epoch=0400-test_mean_score=0.762.ckpt --output_dir data/cp/pusht_4 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/pusht/checkpoints/epoch=0450-test_mean_score=0.747.ckpt --output_dir data/cp/pusht_5 --device cuda:0

### can
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/can/checkpoints/latest.ckpt --output_dir data/cp/can_0 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/can/checkpoints/epoch=0150-test_mean_score=0.820.ckpt --output_dir data/cp/can_1 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/can/checkpoints/epoch=0300-test_mean_score=0.940.ckpt --output_dir data/cp/can_2 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/can/checkpoints/epoch=0350-test_mean_score=0.940.ckpt --output_dir data/cp/can_3 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/can/checkpoints/epoch=0400-test_mean_score=0.980.ckpt --output_dir data/cp/can_4 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/can/checkpoints/epoch=0450-test_mean_score=0.980.ckpt --output_dir data/cp/can_5 --device cuda:0

### lift
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/lift/checkpoints/latest.ckpt --output_dir data/cp/lift_0 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/lift/checkpoints/epoch=0000-test_mean_score=1.000.ckpt --output_dir data/cp/lift_1 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/lift/checkpoints/epoch=0100-test_mean_score=1.000.ckpt --output_dir data/cp/lift_2 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/lift/checkpoints/epoch=0150-test_mean_score=1.000.ckpt --output_dir data/cp/lift_3 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/lift/checkpoints/epoch=0200-test_mean_score=1.000.ckpt --output_dir data/cp/lift_4 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/lift/checkpoints/epoch=0300-test_mean_score=1.000.ckpt --output_dir data/cp/lift_5 --device cuda:0

### square
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/square/checkpoints/latest.ckpt --output_dir data/cp/square_0 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/square/checkpoints/epoch=0100-test_mean_score=0.880.ckpt --output_dir data/cp/square_1 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/square/checkpoints/epoch=0150-test_mean_score=0.800.ckpt --output_dir data/cp/square_2 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/square/checkpoints/epoch=0200-test_mean_score=0.800.ckpt --output_dir data/cp/square_3 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/square/checkpoints/epoch=0250-test_mean_score=0.880.ckpt --output_dir data/cp/square_4 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/square/checkpoints/epoch=0300-test_mean_score=0.820.ckpt --output_dir data/cp/square_5 --device cuda:0

### tool_hang
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/th_1000/checkpoints/epoch=0250-test_mean_score=0.380.ckpt --output_dir data/cp/th_1 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/th_1000/checkpoints/epoch=0300-test_mean_score=0.540.ckpt --output_dir data/cp/th_2 --device cuda:
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/th_1000/checkpoints/epoch=0350-test_mean_score=0.500.ckpt --output_dir data/cp/th_3 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/th_1000/checkpoints/epoch=0400-test_mean_score=0.420.ckpt --output_dir data/cp/th_4 --device cuda:0

    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/tool_hang_cp/checkpoints/latest.ckpt --output_dir data/cp/th_0 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/tool_hang_cp/checkpoints/epoch=0250-test_mean_score=0.340.ckpt --output_dir data/cp/th_1 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/tool_hang_cp/checkpoints/epoch=0300-test_mean_score=0.380.ckpt --output_dir data/cp/th_2 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/tool_hang_cp/checkpoints/epoch=0350-test_mean_score=0.480.ckpt --output_dir data/cp/th_3 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/tool_hang_cp/checkpoints/epoch=0400-test_mean_score=0.420.ckpt --output_dir data/cp/th_4 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/tool_hang_cp/checkpoints/epoch=0450-test_mean_score=0.520.ckpt --output_dir data/cp/th_5 --device cuda:0

### transport
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/transport/checkpoints/latest.ckpt --output_dir data/cp/transport_0 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/transport/checkpoints/epoch=0150-test_mean_score=0.740.ckpt --output_dir data/cp/transport_1 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/transport/checkpoints/epoch=0200-test_mean_score=0.740.ckpt --output_dir data/cp/transport_2 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/transport/checkpoints/epoch=0250-test_mean_score=0.700.ckpt --output_dir data/cp/transport_3 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/transport/checkpoints/epoch=0300-test_mean_score=0.700.ckpt --output_dir data/cp/transport_4 --device cuda:0
    python eval_cp.py --checkpoint /data/wts/consistency_policy_output/ctm/transport/checkpoints/epoch=0350-test_mean_score=0.760.ckpt --output_dir data/cp/transport_5 --device cuda:0

timesteps: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 74, 75, 76, 77, 78, 79], device='cuda:0')
t: 79.9999771118164
next_t: 75.20682525634766
t: 75.20682525634766
next_t: 70.6619644165039
t: 70.6619644165039
next_t: 66.35457611083984
t: 66.35457611083984
next_t: 62.274261474609375
t: 62.274261474609375
next_t: 58.41090774536133
t: 58.41090774536133
next_t: 54.75483322143555
t: 54.75483322143555
next_t: 51.296688079833984
t: 51.296688079833984
next_t: 48.0274658203125
t: 48.0274658203125
next_t: 44.938507080078125
t: 44.938507080078125
next_t: 42.021400451660156
t: 42.021400451660156
next_t: 39.26813507080078
t: 39.26813507080078
next_t: 36.67095947265625
t: 36.67095947265625
next_t: 34.222434997558594
t: 34.222434997558594
next_t: 31.915407180786133
t: 31.915407180786133
next_t: 29.742984771728516
t: 29.742984771728516
next_t: 27.698579788208008
t: 27.698579788208008
next_t: 25.77581214904785
t: 25.77581214904785
next_t: 23.968603134155273
t: 23.968603134155273
next_t: 22.271106719970703
t: 22.271106719970703
next_t: 20.677730560302734
t: 20.677730560302734
next_t: 19.183067321777344
t: 19.183067321777344
next_t: 17.781986236572266
t: 17.781986236572266
next_t: 16.469547271728516
t: 16.469547271728516
next_t: 15.241029739379883
t: 15.241029739379883
next_t: 14.091912269592285
t: 14.091912269592285
next_t: 13.017870903015137
t: 13.017870903015137
next_t: 12.014774322509766
t: 12.014774322509766
next_t: 11.078678131103516
t: 11.078678131103516
next_t: 10.20580005645752
t: 10.20580005645752
next_t: 9.392545700073242
t: 9.392545700073242
next_t: 8.635489463806152
t: 8.635489463806152
next_t: 7.931343078613281
t: 7.931343078613281
next_t: 7.2769975662231445
t: 7.2769975662231445
next_t: 6.6694793701171875
t: 6.6694793701171875
next_t: 6.105963706970215
t: 6.105963706970215
next_t: 5.583760738372803
t: 5.583760738372803
next_t: 5.100316524505615
t: 5.100316524505615
next_t: 4.653203010559082
t: 4.653203010559082
next_t: 4.240114688873291
t: 4.240114688873291
next_t: 3.858869791030884
t: 3.858869791030884
next_t: 3.507390260696411
t: 3.507390260696411
next_t: 3.183716058731079
t: 3.183716058731079
next_t: 2.8859899044036865
t: 2.8859899044036865
next_t: 2.6124563217163086
t: 2.6124563217163086
next_t: 2.361450433731079
t: 2.361450433731079
next_t: 2.131406545639038
t: 2.131406545639038
next_t: 1.9208447933197021
t: 1.9208447933197021
next_t: 1.7283706665039062
t: 1.7283706665039062
next_t: 1.5526701211929321
t: 1.5526701211929321
next_t: 1.3925081491470337
t: 1.3925081491470337
next_t: 1.2467200756072998
t: 1.2467200756072998
next_t: 1.11421537399292
t: 1.11421537399292
next_t: 0.9939699769020081
t: 0.9939699769020081
next_t: 0.8850234746932983
t: 0.8850234746932983
next_t: 0.786477267742157
t: 0.786477267742157
next_t: 0.6974902153015137
t: 0.6974902153015137
next_t: 0.6172764897346497
t: 0.6172764897346497
next_t: 0.5451032519340515
t: 0.5451032519340515
next_t: 0.4802875518798828
t: 0.4802875518798828
next_t: 0.4221931993961334
t: 0.4221931993961334
next_t: 0.3702290952205658
t: 0.3702290952205658
next_t: 0.32384681701660156
t: 0.32384681701660156
next_t: 0.28253674507141113
t: 0.28253674507141113
next_t: 0.245828315615654
t: 0.245828315615654
next_t: 0.21328619122505188
t: 0.21328619122505188
next_t: 0.18450859189033508
t: 0.18450859189033508
next_t: 0.15912558138370514
t: 0.15912558138370514
next_t: 0.13679634034633636
t: 0.13679634034633636
next_t: 0.11720854043960571
t: 0.11720854043960571
next_t: 0.10007573664188385
t: 0.10007573664188385
next_t: 0.08513598144054413
t: 0.08513598144054413
next_t: 0.07215015590190887
t: 0.07215015590190887
next_t: 0.060900572687387466
t: 0.060900572687387466
next_t: 0.051189154386520386
t: 0.051189154386520386
next_t: 0.04283656179904938
t: 0.04283656179904938
next_t: 0.03568049892783165
t: 0.03568049892783165
next_t: 0.02957458421587944
t: 0.02957458421587944
next_t: 0.024387115612626076
t: 0.024387115612626076
next_t: 0.019999999552965164

对scheduler.calc_out()进行修改：

    def calc_out(self, model, trajectory: torch.Tensor, times: torch.Tensor, clamp=False): 
        if isinstance (times, list): # 如果没有reuse，正常执行；有reuse，就取关键步时间来计算
            time = times[0]
        if self.scaling == "boundary":
            # if isinstance (times, list):
            #     c_skip, c_out, c_in = [append_dims(c, trajectory.ndim) for c in self.get_scalings_for_boundary_condition(times[0])]
            # else:
            #     c_skip, c_out, c_in = [append_dims(c, trajectory.ndim) for c in self.get_scalings_for_boundary_condition(times)] # 这几个参数是可以直接根据EDM的表1直接确定的
            c_skip, c_out, c_in = [append_dims(c, trajectory.ndim) for c in self.get_scalings_for_boundary_condition(time)]
        elif self.scaling == "no_boundary":
            c_skip, c_out, c_in = [append_dims(c, trajectory.ndim) for c in self.get_scalings(time)]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

        # if times.ndim > 1:
        #     times = reduce_dims(times, 1)
        rescaled_times = []
        for t in times:       
            rescaled_times.append(1000 * 0.25 * torch.log(t + 1e-44)) # *1000 to make it more stable # 原论文里面是没有1000的
        model_output = model(trajectory * c_in, rescaled_times)

        out = model_output * c_out + trajectory * c_skip
        if clamp: # true
            out = out.clamp(-1.0, 1.0) #this should only happen at inference time

        return out

## 机械臂

pusht输出： 2d x,y 固定z
其他task：10d goal end effector position (3D),goal effector rotation (6-D) and goal gripper position (1D) 最后一个应该是夹爪状态

###  为什么 rot 是 6 个数值？
代码中使用了 6D 旋转表示法 (rotation_6d)，这是一个常用的方式来表示旋转的高维嵌入，具有以下特点：

6D 表示法的定义：

它是通过将 3x3 的旋转矩阵中的前两列拼接起来得到的（2 列 × 3 个分量 = 6 个分量）。
这 6 个分量足以唯一地确定一个旋转矩阵，因为在正交矩阵中，第三列可以通过叉乘前两列计算得到。
优势：

避免了四元数表示的归一化问题（四元数需要满足单位长度约束）。
避免了欧拉角的奇异性问题（例如，万向节锁）。
在优化问题中，6D 表示通常更加稳定，因为它不需要显式归一化。
转换过程：

在 RotationTransformer 中，从其他旋转表示（如轴角、欧拉角或四元数）到 6D 是通过中间的旋转矩阵来实现的。具体函数是 pt.matrix_to_rotation_6d

# 频谱特征分析
 dp的的结构还是原先的先预测噪声再去噪的到去噪后的轨迹；unet是用来预测轨迹的，所以即使输入的信息是上一步去噪后的轨迹，但是整个过程依旧是在预测噪声，解释了为什么我上周画的去噪后的轨迹很奇怪，其实那个是未反归一化的噪声；所以要分析的是encoder各个block的输出 是噪声的什么特征；而不是对于轨迹来说

 所以就不是分析 例如轨迹的高低频特征，而是分析噪声的高低频特征了

 对，但是现在很多都直接去预测去噪后的图像而不是噪声；我推测是傅里叶变换之后频域和图像不一样；可能和通道数有关

 那如果这样的话，最开始做的频谱图就有用了

 在开始decoder的时候，backbone 特征 就是mid block2 的输出，包含的是绝大部分的低频信息；在接受encoder block3 的高频信息之后，在decoder block1的输出包含少量高频信息（为什么是少量我等会讲）

 然后结合encoder block2的跳跃特征，之后在decoder block2 输出非常好的频谱图，四周都是黑的，对应低频信息；但是非常特别的是此时频谱图中心出现狭长亮点，就和图像中心的亮点一样

 （因为维度问题 所以会是狭长亮条；如果对应图像就是中心小区域

 所以我推测，encoder block2应该包含了绝大部分的关键低频信息；因为在decoder block1的输出中心是没有亮点的，所以encoder block的特征可能是部分不关键的高频信息加上部分不关键的低频信息；这样的话才会在频谱图中作用在绝大部分低频区域之上，使整体频谱亮度提升

 所以encoder block3 特征不考虑，对最终实验准确率降低10个点左右，较小；而包含主要高频信息的backbone特征与关键高频信息的encoder block2的特征 不考虑时，实验成功率基本为0 ；影响很大

## 数据集文件夹
并行预训练模型位置：/data/wts/diffusion_policy/data/outputs
2024.11.23:pusht 42 43 44
2024.11.30:transport_mh 43 44 2024.11.19
2024.12.14:transport_ph_ph 43 44 2024.11.11
韬哥服务器：2024.11.25  2024.11.26：square_ph 43 44 已传本服务器

cp 预训练模型：
tool_hang和transport_ph 训练的env runner设置的env_n =28 与dp一致，但是cp的congif中设置的都是1 ;can与tool
只有tool_hang模型是用torch.dataparallel跑的，但是也是在单卡上跑的，评估的时候要额外代码； 
/data/wts/consistency_policy_output/edm/tool_hang

square_ph不知道是不是； /data/wts/consistency_policy_output/checkpoint/edm
lift can 在本服务器上
transport_ph

cp 训练：
can lift square transport  pusht 

## 关键步设置：
先运行一遍特征分析，可以得到
edm tool_hang：
key steps : [0, 10, 11, 12, 18, 26, 60, 61, 62, 63]
final score: 0.00015542245285359968
key steps : [0, 9, 10, 11, 17, 25, 59, 60, 61, 78]
final score: 0.013663251569187195
key steps : [0, 10, 11, 12, 18, 26, 32, 60, 61, 62]
final score: 0.00017373829859934628
key steps : [0, 9, 10, 11, 25, 31, 59, 60, 61, 78]
final score: 0.013700962026778141

key steps : [0, 10, 11, 12, 18, 60, 61, 62]
final score: 0.00012869796088580185
key steps : [0, 9, 10, 11, 59, 60, 61, 78]
final score: 0.013636275842045506
key steps : [0, 10, 11, 12, 26, 60, 61, 62]
final score: 0.0001448564375778005
key steps : [0, 10, 11, 25, 59, 60, 61, 78]
final score: 0.013671246602007157

edm_can
key steps : [0, 3, 4, 12, 14, 17, 19, 25, 26, 27]
final score: 0.06988771946169435
key steps : [0, 2, 3, 11, 13, 16, 18, 25, 26, 78]
final score: 0.3902103085187264
key steps : [0, 3, 4, 12, 14, 17, 19, 26]
final score: 0.05790950293652713
key steps : [0, 2, 3, 11, 13, 16, 25, 78]

edm_pusht
key steps : y [0, 7, 9, 12, 17, 25, 28, 31, 32, 34]
next_y[0, 6, 8, 11, 16, 24, 27, 31, 33, 78]
final score: 0.001254711503643193
key steps : [0, 7, 9, 12, 17, 25, 28, 34]
final score: 0.0009987302233639638
key steps : [0, 6, 8, 11, 16, 24, 27, 78]
final score: 0.004122984720743262

key steps : [0, 3, 9, 12, 17, 19, 26, 28]
final score: 0.0009099913197330513
key steps : [0, 2, 8, 11, 16, 18, 27, 78]
final score: 0.0047482661869766634

key steps : [0, 5, 6]
final score: 0.006491557603572
key steps : [0, 5, 6]
final score: 0.010558019323891915

edm_lift
key steps : [0, 1, 10, 14, 15, 17, 18, 22]
final score: 0.04827588803600519
key steps : [0, 9, 13, 14, 16, 17, 77, 78]
final score: 0.6770106988679617
key steps : [0, 1, 2, 10, 13, 15, 18, 19, 20, 21]
final score: 0.07849177432013675
key steps : [0, 1, 9, 12, 14, 17, 18, 19, 20, 78]
final score: 0.6684874604153446

pusht：[0, 8, 9]
square_image_ph：[0, 8, 9]
square_image_mh: [0, 8, 9]
transport_image_ph: [0, 8, 9]
transport_image_mh: [0, 8, 9]
tool_hang_ph: [0, 8, 9]

[2] 3483411
[1] 1663188
[1] 2374824
too_hang_ph 43 44[1] 3043941 
[1] 2081774
[1] 311800
[1] 367140
[1] 519014
square_mh 43 [1] 3245555  44 [2] 3252624
[1] 3902227

    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/edm/lift/checkpoints/latest.ckpt --output_dir data/lift_edm_eval_output --device cuda:0
    python consistency_eval.py --checkpoint /data/wts/consistency_policy_output/checkpoint/edm/latest.ckpt --output_dir data/square_edm_eval_output --device cuda:0 


## 补偿系数设置
使用去噪过程中的能量变化差异作为放缩比

        low_energy_list = []
        high_energy_list = []

        model_output, low_energy, high_energy = model(trajectory, t, 
            local_cond=local_cond, global_cond=global_cond, flag = flag)
        low_energy_list.append(low_energy)
        high_energy_list.append(high_energy) 

        low_cc = [math.sqrt(low_energy_list[idx+1]/low_energy_list[idx]) for idx in range(len(low_energy_list)-1)]
        high_cc = [math.sqrt(high_energy_list[idx+1]/high_energy_list[idx]) for idx in range(len(high_energy_list)-1)]
        print(f"low energy compensation coefficient{low_cc}")
        print(f"high energy compensation coefficient{high_cc}")

pusht 
low energy compensation coefficient[1.0065590212174718, 0.9719612672058621, 0.9404256311568802, 0.9232494001306547, 0.891076559067838, 0.8634302347314536, 0.8293282745928686, 0.7548860498040626, 0.648648997365932]
high energy compensation coefficient[0.9668299197319893, 0.9490387671363641, 0.9262911523076228, 0.9010784310721689, 0.8630548585957734, 0.8153314771158023, 0.7460979841022282, 0.6232315316002098, 0.41715442797841384]


# edm-faster 修改

    [249]while curr_step<all_steps-1: # 在这里进行扩散    # 80步的话最多到79步就停了，因为可以通过79步计算出80步的结果
    [262]    if curr_step<all_steps-1:

    for idx, (times, next_times) in enumerate(zip(t, next_t)):
        step = append_dims((next_times - times), dims) # 维度对齐 此时的得到得一组list
        noise_y = denoisedy[idx*bs_perstep:(idx+1)*bs_perstep]
        # dy = (y - denoisedy) / append_dims(t, dims)
        dy = (curr_y - noise_y) / append_dims(times, dims)
        
        y_next = samples + step * dy
        # 将第一次的y_next的噪声reuse，和encoder reuse本质上没有区别 每一次reuse最后一步重新计算提高准确率 10+20/80+80
        if denoisedy_next is None or idx == len(t) - 1:
            denoisedy_next = self.calc_out(model, y_next, next_times, clamp = clamp) 
        else:
            continue

## ckpt键值修改

    checkpoint = torch.load(cfg.policy.teacher_path) 
    def recursive_modify(value):
        if isinstance(value, dict):
            new_value = {}
            for sub_key, sub_value in value.items():
                # 修改嵌套键值（例如移除 'module.' 前缀）
                if isinstance(sub_key, str) and sub_key.startswith('module.'):
                    new_sub_key = sub_key[7:]
                else:
                    new_sub_key = sub_key
                new_value[new_sub_key] = recursive_modify(sub_value)  # 递归修改
            return new_value
        elif isinstance(value, list):
            return [recursive_modify(item) for item in value]  # 如果是列表，递归修改
        else:
            return value  # 原样返回非字典/列表的值
        
    # 更新检查点中的 state_dict 
    for key, value in checkpoint['state_dicts'].items():
        checkpoint['state_dicts'][key] = recursive_modify(value)
        
    new_checkpoint_path = "/data/wts/consistency_policy_output/edm/tool_hang/checkpoints/latest_new.ckpt"  # 替换为保存的新路径
    torch.save(checkpoint, new_checkpoint_path)
    
    cfg.policy.teacher_path = new_checkpoint_path
    cfg.policy.edm = new_checkpoint_path
    
    pdb.set_trace()
    checkpoint = torch.load(cfg.policy.teacher_path)
    for key, value in checkpoint['state_dicts'].items():
        pass

# real
[1] 2763355 pap
pusht 在涛哥服务器上

# cp
can [1] 2943759
lift [2] 2949124
pusht [3] 2966519
th [1] 19818

# 绘图
net_feature文件夹下
启动环境：faster_diffusion_policy
运行eval.py代码即可运行。注意修改绘图代码中文件的名称与图像存储位置

时间步之间特征差异：横轴要画出对应的时间步。纵轴统一，是否要除以batchsize