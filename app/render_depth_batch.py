import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse
import shutil
import json
from tqdm import tqdm
import random
import torch

from hair_renderer import *

import multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

def render_depth(split):
    hair_dir = './data/hair_ply/'#hair_ply_1024
    orien2d_dir = './data/render/RENDER_ORIEN/'
    camera_dir = './data/render/PARAM/'
    output_dir = './data/render/RENDER_DEPTH_SOFT_10k/'#

    # create renderer with SoftRas
    # rasterizer = SoftRasterizer(image_size=512, near=-100, far=100, sigma_val=1e-5)
    rasterizer = SoftRasterizer(image_size=512, near=-100, far=100, sigma_val=1e-6, gamma_val=1e-6, eps=1e-6)

    hair_path = hair_dir + split.split('/')[0] + '.ply'
    camera_path = camera_dir + split[:-3] + 'npy'
    output_path = output_dir + split[:-3] + 'png' #.split('/')[0] + '_' + split.split('/')[1][:-4] + '.png'

    os.makedirs(output_dir + split.split('/')[0], exist_ok=True)
    if os.path.exists(output_path):
        return

    # load mesh and transform to camera space
    render_depth = RenderDepth(hair_path, camera_path)
    mesh = render_depth.forward()

    images = rasterizer(mesh)

    #mask
    images = images[:,:3]*images[:,3]

    images = torch.flip(images, [2])
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    image = (255*image).astype(np.uint8)
    imageio.imwrite(output_path, image)

if __name__ == '__main__':
    print('render hair depth 10k 3/3')
    mp.set_start_method('spawn')
    # mode = 'train'
    # split_file = '../Hair-PIFu-norm/data/render_'+mode+'_split_00101_bust.json'
    # with open(split_file,'r') as f:
    #     splits = json.load(f)
    splits = ['strands00075_00367_00101/5_3_00.npy', 'strands00075_00367_00101/330_12_00.npy', 'strands00075_00367_00101/350_351_00.npy']

    # splits = splits[:10]
    # splits = splits[0::3]
    # splits = splits[1::3]
    # splits = splits[2::3]

    p = mp.Pool(processes=6)
    p.map(render_depth, splits)
    p.close()
    p.join()
