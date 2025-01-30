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

def render_orien(hair_path, camera_path, mask_path, output_path, output_with_seg_path):
    # create renderer with SoftRas
    # rasterizer = SoftRasterizer(image_size=512, near=-100, far=100, sigma_val=1e-5)
    rasterizer = SoftRasterizer(image_size=512, near=-100, far=100, sigma_val=1e-6, gamma_val=1e-6, eps=1e-6)

    if os.path.exists(output_path):
        return

    # load mesh and transform to camera space
    render_orien = RenderOrien(hair_path, camera_path)
    mesh = render_orien.forward()

    images = rasterizer(mesh)

    #mask
    images = images[:,:3]*images[:,3]

    images = torch.flip(images, [2])
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    image = (255*image).astype(np.uint8)
    imageio.imwrite(output_path, image)

    mask = np.array(imageio.imread(mask_path))/255
    if len(mask.shape)==2:
        mask = np.repeat(mask[:,:,None], 3, axis=2)
    if len(mask.shape)==3:
        mask = mask[:,:,:3]
    mask_bool = (mask>0.5)
    imageio.imwrite(output_with_seg_path, image*mask_bool)


if __name__ == '__main__':
    name = 'orien_orien2d10k_directed_lr-4_cano_modvisi_3view_NormDepthR_noZ_noImgNorm_pseudo_weak_label'

    root = '/data2/paul/hair/Dataset/real_img_dataset/All/'
    split_file = '/data2/paul/hair/Dataset/real_img_dataset/All/split_test_196.json'

    with open(split_file,'r') as f:
        items = json.load(f)

    for item in tqdm(items):
        # if item[-3:]!='obj':
        #     continue

        hair_path = '/data2/paul/hair/Dataset/real_img_dataset/All/test_results/' + name + '/' + item[:-3] + 'ply'
        camera_path = root+'param/' + item[:-3] + 'npy'
        mask_path = root+'seg/' + item[:-3] + 'png'
        output_path = root + 'test_results/rendered_orien/' + name + '/'  + item[:-3] + 'png'
        output_with_seg_path = root + 'test_results/rendered_orien_masked/' + name + '/'  + item[:-3] + 'png'

        os.makedirs(root + 'test_results/rendered_orien/' + name + '/', exist_ok=True)
        os.makedirs(root + 'test_results/rendered_orien_masked/' + name + '/', exist_ok=True)

        render_orien(hair_path, camera_path, mask_path, output_path, output_with_seg_path)

# CUDA_VISIBLE_DEVICES=2 screen python -m app.render_orien_from_prediction