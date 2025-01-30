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

os.environ["OMP_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["VECLIB_MAXIMUM_THREADS"] = "5"
os.environ["NUMEXPR_NUM_THREADS"] = "5"

def render_hair(split):
    hair_dir = './data/hair_ply/'#hair_ply_1024
    orien2d_dir = './data/render/RENDER_ORIEN/'
    camera_dir = './data/render/PARAM/'
    output_dir = './data/render/RENDER_ORIEN_SOFT_10k_DIRECTED/'#

    rasterizer = SoftRasterizer(image_size=512, near=-100, far=100, sigma_val=1e-6, gamma_val=1e-6, eps=1e-6)

    hair_path = hair_dir + split.split('/')[0] + '.ply'
    orien2d_path = orien2d_dir + split
    camera_path = camera_dir + split[:-3] + 'npy'
    output_path = output_dir + split[:-3] + 'png' #.split('/')[0] + '_' + split.split('/')[1][:-4] + '.png'

    os.makedirs(output_dir + split.split('/')[0], exist_ok=True)
    if os.path.exists(output_path):
        return

    # load mesh and transform to camera space
    render_mesh = RenderOrien(hair_path, camera_path)
    mesh = render_mesh.forward()

    images = rasterizer(mesh)

    #mask
    images = images[:,:3]*images[:,3]

    images = torch.flip(images, [2])
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    image = (255*image).astype(np.uint8)
    imageio.imwrite(output_path, image)

    # create renderer with SoftRas
    # rasterizer = SoftRasterizer(image_size=512, near=-100, far=100, sigma_val=1e-5)
    

if __name__ == '__main__':
    print('render hair orien 10k 3/3')
    mp.set_start_method('spawn')
    # mode = 'train'
    # split_file = '../Hair-PIFu-norm/data/'+mode+'_split_00101_bust.json'
    # with open(split_file,'r') as f:
    #     splits = json.load(f)

    # mode = 'train'
    # split_file = '../Hair-PIFu-norm/data/'+mode+'_split_00101_bust.json'
    # with open(split_file,'r') as f:
    #     train_splits = json.load(f)

    # splits = splits + train_splits

    splits = ['strands00075_00367_00101/5_3_00.npy', 'strands00075_00367_00101/330_12_00.npy', 'strands00075_00367_00101/350_351_00.npy']
    # splits = splits[0::3]
    # splits = splits[1::3]
    # splits = splits[2::3]

    # splits.reverse()

    # for split in tqdm(splits):
        # render_hair(split)
    p = mp.Pool(processes=5)
    p.map(render_hair, splits)
    p.close()
    p.join()
