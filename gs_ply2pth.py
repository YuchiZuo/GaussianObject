import os
import sys
import uuid
import json
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Optional

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.functional.regression import pearson_corrcoef
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss,l2_loss, ssim, monodisp
from utils.pose_utils import update_pose, get_loss_tracking
from torch.utils.tensorboard.writer import SummaryWriter

def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    gaussians = GaussianModel(3)
    gaussians.training_setup(opt)
    ply_path = "./data/R63_three_view/model_v1.ply"
    gaussians.load_ply(ply_path)
    iteration = 1
    model_path = "./data/R63_three_view"
    torch.save((gaussians.capture(), iteration), "/ckpt" + str(iteration) + ".pth")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    ### some exp args
    parser.add_argument("--sparse_view_num", type=int, default=-1, 
                        help="Use sparse view or dense view, if sparse_view_num > 0, use sparse view, \
                        else use dense view. In sparse setting, sparse views will be used as training data, \
                        others will be used as testing data.")
    parser.add_argument("--use_mask", default=True, help="Use masked image, by default True")
    parser.add_argument('--use_dust3r', action='store_true', default=False,
                        help='use dust3r estimated poses')
    parser.add_argument('--dust3r_json', type=str, default=None)
    parser.add_argument("--init_pcd_name", default='origin', type=str, 
                        help="the init pcd name. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument("--transform_the_world", action="store_true", help="Transform the world to the origin")
    parser.add_argument('--mono_depth_weight', type=float, default=0.0005, help="The rate of monodepth loss")
    parser.add_argument('--lambda_t_norm', type=float, default=0.0005)
    parser.add_argument('--mono_loss_type', type=str, default="mid")
    parser.add_argument('--w_pose', action='store_true', default=False,
                        help='use diff-gaussian-rasterization-w-pose ')   
    parser.add_argument('--freeze_pos', action='store_true', default=False,
                        help='freeze_pos')      
    parser.add_argument('--normal_loss', action='store_true', default=False,
                        help='use normal_loss') 
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")