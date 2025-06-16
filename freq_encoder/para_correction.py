def ParaCorrection(cfg, 
                   step_num=None, 
                   dataset_path=None, 
                   batch_size=64, 
                   scheduler="DDPM",
                   n_envs=None 
                   ):
    if cfg.dataloader.batch_size != batch_size:
        cfg.dataloader.batch_size = batch_size
        cfg.val_dataloader.batch_size = batch_size
        
    # dp square_ph edm
    if dataset_path is not None:
        cfg.task.dataset.dataset_path = dataset_path
        cfg.task.dataset_path = dataset_path
        cfg.task.env_runner.dataset_path = dataset_path

    # policy must change by yourself
    if scheduler is not None:
        if scheduler == "DDIM":
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
            if step_num is not None:
                cfg.policy.num_inference_steps = step_num
        
        elif scheduler == "EDM":
            if step_num is not None:
                cfg.policy.noise_scheduler.bins = step_num
            
    if n_envs is not None:
        cfg.task.env_runner.n_envs = n_envs