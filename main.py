import os
import pdb
import cv2
import math
import random
import shutil
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import SimpleNVP, RST_NVP, T_NVP
from renderer import ellipse_renderer, capsule_renderer
from utils import get_geodesic_dist, get_euclidean_dist, chamfer_loss, preprocess_image

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='KnotArt')
    #### Optimization parameters
    parser.add_argument('--img_path', type=str, default='data/Image1.png', help='target image path')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--random_seed', type=int, default=random.randint(1, 10000), help='random seed')
    parser.add_argument('--num_epochs', type=int, default=25000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    #### Camera and tube parameters
    parser.add_argument('--fx', type=float, default=1, help='camera focal length in x')
    parser.add_argument('--fy', type=float, default=1, help='camera focal length in y')
    parser.add_argument('--fz', type=float, default=2, help='camera focal length in z')
    parser.add_argument('--z_near', type=float, default=1e-3, help='camera near clipping plane')
    parser.add_argument('--z_far', type=float, default=1e3, help='camera far clipping plane')
    parser.add_argument('--rho', type=float, default=0.05, help='radius of tube')
    #### Renderer Parameters
    parser.add_argument('--renderer_type', type=str, default='ellipse', help='type of renderer', choices=['ellipse', 'capsule'])
    parser.add_argument('--tau', type=float, default=1e4, help='renderer hardness factor')
    parser.add_argument('--tau_ellipse', type=float, default=1e3, help='renderer hardness factor for ellipses')
    parser.add_argument('--tau_quadrilateral', type=float, default=1e2, help='renderer hardness factor for quadrilaterals')
    parser.add_argument('--temp1_max', type=float, default=1e5, help='max limit for absolute value of temp1')
    parser.add_argument('--temp2_max', type=float, default=1e5, help='max limit for absolute value of temp2')
    parser.add_argument('--temp3_max', type=float, default=1e5, help='max limit for absolute value of temp3')
    parser.add_argument('--eps_z_pos', type=float, default=1e-2, help='min limit for z coordinate of points')
    parser.add_argument('--eps_render', type=float, default=1e-8, help='epsilon min value for rendering')
    parser.add_argument('--discriminant_sq_min', type=float, default=1e-4, help='min limit for discriminant')
    parser.add_argument('--alpha_max', type=float, default=1e5, help='max limit for absolute value of alpha')
    parser.add_argument('--beta_max', type=float, default=1e5, help='max limit for absolute value of beta')
    parser.add_argument('--gamma_max', type=float, default=1e5, help='max limit for absolute value of gamma')
    #### Additional parameters
    parser.add_argument('--lamb', type=float, default=0.9, help='lambda factor for uniformity of intervals')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for numerical stability')
    #### Sample sizes
    parser.add_argument('--num_samp', type=int, default=3500, help='number of points to sample on the knot')
    parser.add_argument('--num_dist', type=int, default=10000, help='number of pairs sampled for loss computation')
    parser.add_argument('--img_size', type=int, default=256, help='size of rendered image')
    #### logging, saving and checkpointing frequency
    parser.add_argument('--viz_period', type=int, default=1, help='frequency of visualizing renders')
    parser.add_argument('--print_period', type=int, default=1, help='frequency of printing losses')
    parser.add_argument('--ckpt_period', type=int, default=1000, help='frequency of saving checkpoints')
    #### Loss weights and maximum allowed values
    parser.add_argument('--w_img', type=float, default=1, help='weight for image loss')
    parser.add_argument('--w_rep', type=float, default=0, help='weight for mobius loss')
    parser.add_argument('--w_len', type=float, default=0, help='weight for length loss')
    parser.add_argument('--w_occ', type=float, default=0, help='weight for constraint region occupancy loss')
    parser.add_argument('--w_bend', type=float, default=0, help='weight for bending loss')
    parser.add_argument('--L_max', type=float, default=1e3, help='maximum allowed length of knot')
    parser.add_argument('--B_max', type=float, default=1e-2, help='maximum allowed bending in knot')
    #### INN parameters
    parser.add_argument('--NVP_type', type=str, default='T_NVP', help='INN type', choices=['SimpleNVP', 'RST_NVP', 'T_NVP'])
    parser.add_argument('--num_layers', type=int, default=8, help='number of layers in INN')
    parser.add_argument('--hidden_size', type=int, default=1024, help='number of hidden units in each layer of INN')




    args = parser.parse_args()

    #### Optimization parameters
    img_path = args.img_path
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    num_epochs = args.num_epochs
    lr = args.lr
    #### Camera and tube parameters
    fx = args.fx
    fy = args.fy
    fz = args.fz
    z_near = args.z_near
    z_far = args.z_far
    rho = args.rho
    #### Renderer Parameters
    renderer_type = args.renderer_type
    tau = args.tau
    tau_ellipse = args.tau_ellipse
    tau_quadrilateral = args.tau_quadrilateral
    temp1_max = args.temp1_max
    temp2_max = args.temp2_max
    temp3_max = args.temp3_max
    eps_z_pos = args.eps_z_pos
    eps_render = args.eps_render
    discriminant_sq_min = args.discriminant_sq_min
    alpha_max = args.alpha_max
    beta_max = args.beta_max
    gamma_max = args.gamma_max
    #### Additional parameters
    lamb = args.lamb
    eps = args.eps
    #### Sample sizes
    num_samp = args.num_samp
    num_dist = args.num_dist
    img_size = args.img_size
    #### logging, saving and checkpointing frequency
    viz_period = args.viz_period
    print_period = args.print_period
    ckpt_period = args.ckpt_period
    #### Loss weights and maximum allowed values
    w_img = args.w_img
    w_rep = args.w_rep
    w_len = args.w_len
    w_occ = args.w_occ
    w_bend = args.w_bend
    L_max = args.L_max
    B_max = args.B_max
    #### INN parameters
    NVP_type = args.NVP_type
    num_layers = args.num_layers
    hidden_size = args.hidden_size

    #### Set random seed for reproducibility and device for computation
    print("Random Seed: ", random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device("cuda:%d"%(gpu_id) if (torch.cuda.is_available()) else "cpu")

    #### Define the required directories
    img_name = os.path.splitext(os.path.normpath(img_path).split(os.path.sep)[-1])[0]
    exp_path = '{}_{}'.format(img_name,random_seed)
    log_path = os.path.join(exp_path, 'logs')
    res_path = os.path.join(exp_path, 'results')
    ckpt_path = os.path.join(exp_path, 'ckpt')

    #### Create the required directories
    os.makedirs(exp_path,exist_ok=True)
    os.makedirs(log_path,exist_ok=True)
    os.makedirs(res_path,exist_ok=True)
    os.makedirs(ckpt_path,exist_ok=True)

    #### Initialiate logging file
    logger = open(os.path.join(log_path, 'logs.txt'),'a')
    logger.write("\nRandom Seed: {}\n\n".format(random_seed))
    logger.close()

    #### Camera Rotation and Translation
    INV_ROT =  torch.tensor([   [   [ 1, 0, 0],
                                    [ 0, 1, 0],
                                    [ 0, 0, 1]],
                                    ]).unsqueeze(1).type(torch.float).to(device)                                # 3 x 1 x 3 x 3
    INV_TRANS = -1 * torch.tensor([ [   [0], 
                                        [0], 
                                        [0]], 
                                        ]).unsqueeze(1).type(torch.float).to(device)                            # 3 x 1 x 3 x 1

    #===> Region inside which the tube is constrained to be <===#
    constraint_region = torch.tensor([  [      0,    0,    1, -z_near], 
                                        [      0,    0,   -1,   z_far],
                                        [      0,  -fz,   fy,       0],
                                        [     fz,    0,   fx,       0],
                                        [      0,   fz,   fy,       0],
                                        [    -fz,    0,   fx,       0],
                                        ]).type(torch.float32).unsqueeze(0).to(device)                          # 1 x 6 x 4
    constraint_region = constraint_region/(constraint_region[...,:-1]).square().sum(-1,keepdims=True)

    #### Image Pixel Grid on camera plane
    grid_x = torch.linspace(-fx, fx, img_size)                                                                  # Ni
    grid_y = torch.linspace(-fy, fy, img_size)                                                                  # Ni
    pixels = torch.cartesian_prod(grid_x, grid_y)                                                               # M x 2 , where M = Ni*Ni   
    pixels = pixels.unsqueeze(0).to(device)                                                                     # 1 x M x 2
    pixel_aug = torch.cat([pixels, fz*torch.ones((1,img_size**2, 1),device=device)], -1)                        # 1 x M x 3
    pixel_norm_sq = pixel_aug.square().sum(-1)                                                                  # 1 x M

    #### Initialize figure for visualization
    fig = plt.figure()
    
    #### Initialize homeomorphism network
    homeo = eval(NVP_type)(n_layers = num_layers, hidden_size = hidden_size, device=device).to(device)
    
    #### Initialize optimizer
    opt = torch.optim.Adam(list(homeo.parameters()), lr=lr)

    #### Load and preprocess target image
    image = preprocess_image(img_path, img_size, device)

    #### The main optimization loop
    for i in range(num_epochs):

        #### Zero out gradients
        opt.zero_grad()

        #### Sample Points on the template ring
        shift = 2 * math.pi * torch.rand(1)                                                                     # 1
        intvls = torch.nn.Softmax(0)(lamb + (1-lamb)*torch.rand(num_samp))                                      # Ns
        end = torch.cumsum(intvls,0)                                                                            # Ns
        start = end - intvls                                                                                    # Ns
        theta = 2 * math.pi * (start + shift)                                                                   # Ns
        ring = torch.stack([torch.cos(theta),torch.sin(theta), fz + torch.ones_like(theta)],-1).to(device)      # Ns x 3

        #### Forward Pass to get points on knot
        points = homeo.forward(ring)                                                                            # Ns x 3
        
        #### Points in Camera coordinate frame
        points_cam = INV_ROT[:1].matmul(points.unsqueeze(-1).unsqueeze(0) + INV_TRANS[:1]).squeeze(-1)          # 1 x Ns x 3

        #### Compute Length Loss
        lens = (points - points.roll(1,0)).norm(dim=-1)                                                         # 3 x Ns
        total_length = lens.sum()                                                                               # scalar
        len_loss = torch.nn.functional.relu(total_length - L_max)                                               # scalar    

        #### Compute Euclidean and Geodesic Distances
        idx = torch.randint(low=0,high=num_samp,size=(num_dist,2))                                              # Nd x 2
        csum = lens.cumsum(0)                                                                                   # Ns
        total_loop_length = csum[-1]                                                                            # scalar
        geo_dist = get_geodesic_dist(csum, idx[:,0], idx[:,1])                                                  # Nd
        euc_dist = get_euclidean_dist(points, idx[:,0], idx[:,1])                                               # Nd
        
        #### Compute Mobius Loss
        nz_idx = idx[:,0]!=idx[:,1]                                                                             # Nd
        mobius_energy = (1/((euc_dist[nz_idx]).square() + eps)) - (1/((geo_dist[nz_idx]).square() + eps ))      # Nnz
        mob_loss = (mobius_energy * torch.maximum(2*rho - euc_dist[nz_idx], 
                            torch.zeros_like(nz_idx[nz_idx]).type(torch.float32).to(device))).mean()            # scalar

        #### Compute Constraint Region Occupancy Loss
        homo_point = torch.cat((points_cam, torch.ones(1,num_samp,1).to(device)), -1).unsqueeze(-1)             # 1 x Ns x 4 x 1
        occ = constraint_region.unsqueeze(0).matmul(homo_point).squeeze(-1)                                     # 1 x Ns x 6        
        occ_loss = torch.nn.functional.relu((-occ + rho)).mean()                                                # scalar
        
        #### Compute Bending Loss
        neigh_len = int(0.01*num_samp)                                                                          # scalar
        fwd_diff = torch.nn.functional.normalize(points.roll(-1,0) - points,dim=-1)                             # Ns x 3
        bwd_diff = torch.nn.functional.normalize(points - points.roll(1,0),dim=-1)                              # Ns x 3
        bending = 0.5*(1-torch.multiply(fwd_diff, bwd_diff).sum(-1) )                                           # Ns
        bend_csum = bending.cumsum(0)                                                                           # Ns
        idx_low = torch.remainder(torch.arange(num_samp)-neigh_len, num_samp).to(device)                        # Ns
        idx_high = torch.remainder(torch.arange(num_samp)+neigh_len, num_samp).to(device)                       # Ns
        neigh_bend = bend_csum[idx_high] - bend_csum[idx_low]                                                   # Ns
        neigh_bend = torch.where(idx_high>idx_low, neigh_bend, bend_csum[-1] - neigh_bend)                      # Ns
        bend_loss = torch.nn.functional.relu(neigh_bend - B_max).mean()                                         # scalar        

        #### Render Images and Compute Image Loss
        P = points_cam[0]                                                                                       # Ns x 3
        if renderer_type == 'ellipse':
            I_rendered = ellipse_renderer(P, pixel_aug, pixel_norm_sq, rho, img_size, tau)                      # Ni x Ni
        elif renderer_type == 'capsule':
            I_rendered = capsule_renderer(P, pixels, rho, img_size, tau_ellipse, tau_quadrilateral, fz, 
                                            temp1_max, temp2_max, temp3_max, eps_z_pos, eps_render,
                                            discriminant_sq_min, alpha_max, beta_max, gamma_max)                # Ni x Ni
        image_loss = (I_rendered - image).square().sum()/(img_size**2)                                          # scalar


        #### Total Loss 
        loss = w_img*image_loss + w_rep*mob_loss + w_len*len_loss + w_occ*occ_loss + w_bend*bend_loss           # scalar

        #### compute gradients and update parameters 
        loss.backward()
        opt.step()

        #### Logging
        log_string = '{}/{} | Img_Loss: {:.4f}  | Mob_Loss: {:.4f} | Occ_Loss: {:.4f} | Len_Loss: {:.4f} | Bend_Loss: {:.4f}'.format(i+1,num_epochs, image_loss, mob_loss, occ_loss, len_loss, bend_loss)
        logger = open(os.path.join(log_path, 'logs.txt'),'a')
        logger.write(log_string+'\n')
        logger.close()

        #### Print losses
        if i%print_period==0:
            print(log_string)

        #### Save rendered image
        if i%viz_period==0:
            I_save = 255*(1 - torch.cat([image,I_rendered],-1)).detach().cpu().numpy()
            im = Image.fromarray(I_save).convert("RGB")
            im.save(os.path.join(res_path, 'result_{:d}.png'.format(i)))
        
        #### Save checkpoint
        if i%ckpt_period==0:
            torch.save({
                'iter': i,
                'homeo_state_dict': homeo.state_dict(),
                'opt_state_dict': opt.state_dict()}, 
                    os.path.join(ckpt_path, '{:d}.pth'.format(i)))


# #### Load checkpoint
# ckpt = torch.load(os.path.join(ckpt_path, '{:d}.pth'.format(i)))
# i = ckpt['iter']
# homeo.load_state_dict(ckpt['homeo_state_dict'])
# opt.load_state_dict(ckpt['opt_state_dict'])



