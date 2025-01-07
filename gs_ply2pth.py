from scene import Scene, GaussianModel
import torch

gaussians = GaussianModel(3)
ply_path = "./data/R63_three_view/model_v1.ply"
gaussians.load_ply(ply_path)
iteration = 1
model_path = "./data/R63_three_view"
torch.save((gaussians.capture(), iteration), "/ckpt" + str(iteration) + ".pth")

