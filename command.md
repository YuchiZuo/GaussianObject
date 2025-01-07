usage: train_gs.py [-h] [--sh_degree SH_DEGREE] [--source_path SOURCE_PATH] [--model_path MODEL_PATH] [--images IMAGES] [--resolution RESOLUTION] [--white_background]
                   [--data_device DATA_DEVICE] [--eval] [--max_num_splats MAX_NUM_SPLATS] [--iterations ITERATIONS] [--position_lr_init POSITION_LR_INIT] [--position_lr_final POSITION_LR_FINAL]  
                   [--position_lr_delay_mult POSITION_LR_DELAY_MULT] [--position_lr_max_steps POSITION_LR_MAX_STEPS] [--feature_lr FEATURE_LR] [--opacity_lr OPACITY_LR]
                   [--scaling_lr SCALING_LR] [--rotation_lr ROTATION_LR] [--percent_dense PERCENT_DENSE] [--lambda_dssim LAMBDA_DSSIM] [--lambda_silhouette LAMBDA_SILHOUETTE]
                   [--densification_interval DENSIFICATION_INTERVAL] [--opacity_reset_interval OPACITY_RESET_INTERVAL] [--remove_outliers_interval REMOVE_OUTLIERS_INTERVAL]
                   [--densify_from_iter DENSIFY_FROM_ITER] [--densify_until_iter DENSIFY_UNTIL_ITER] [--densify_grad_threshold DENSIFY_GRAD_THRESHOLD]
                   [--start_sample_pseudo START_SAMPLE_PSEUDO] [--end_sample_pseudo END_SAMPLE_PSEUDO] [--sample_pseudo_interval SAMPLE_PSEUDO_INTERVAL] [--random_background]
                   [--pose_iterations POSE_ITERATIONS] [--convert_SHs_python] [--compute_cov3D_python] [--debug] [--ip IP] [--port PORT] [--debug_from DEBUG_FROM] [--detect_anomaly]
                   [--test_iterations TEST_ITERATIONS [TEST_ITERATIONS ...]] [--save_iterations SAVE_ITERATIONS [SAVE_ITERATIONS ...]] [--quiet]
                   [--checkpoint_iterations CHECKPOINT_ITERATIONS [CHECKPOINT_ITERATIONS ...]] [--start_checkpoint START_CHECKPOINT] [--sparse_view_num SPARSE_VIEW_NUM] [--use_mask USE_MASK]     
                   [--use_dust3r] [--dust3r_json DUST3R_JSON] [--init_pcd_name INIT_PCD_NAME] [--transform_the_world] [--mono_depth_weight MONO_DEPTH_WEIGHT] [--lambda_t_norm LAMBDA_T_NORM]      
                   [--mono_loss_type MONO_LOSS_TYPE]

& E:/Conda/gs_cu121/python.exe d:/Desk/deepLearning/work/3dgs/modified/GaussianObject/train_gs.py --source_path data/R63_three_view  --model_path output/fix_intrinsics --resolution 1 --white_background --sparse_view_num 3 --sh_degree 2 --init_pcd_name points3D --save_iterations 1 500 2000 5000

& E:/Conda/gs_cu121/python.exe d:/Desk/deepLearning/work/3dgs/modified/GaussianObject/train_gs.py --source_path data/R63_three_view  --model_path output/v2_w_pose --resolution 1 --white_background --sparse_view_num 3 --sh_degree 3 --init_pcd_name points3D --save_iterations 1 500 2000 5000

& E:/Conda/gs_cu121/python.exe d:/Desk/deepLearning/work/3dgs/modified/GaussianObject/train_gs.py --source_path data/R63_three_view   --resolution 1 --white_background --sparse_view_num 3 --save_iterations 1 500 2000 5000 --init_pcd_name points3D --model_path output/woPose_freezeXYZ_sh0_voxel0.03 --sh_degree 0 