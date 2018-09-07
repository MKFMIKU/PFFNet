#!/usr/bin/env python
import argparse
import utils
from PIL import Image
import numpy as np
import scipy.misc


parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--data", type=str, default="output", help="path to load data images")
parser.add_argument("--gt", type=str, help="path to load gt images")

opt = parser.parse_args()
print(opt)

datas = utils.load_all_image(opt.data)
gts = utils.load_all_image(opt.gt)

datas.sort()
gts.sort()

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

psnrs = []
for i in range(len(datas)):
    data = scipy.misc.fromimage(Image.open(datas[i])).astype(float)/255.0
    gt = scipy.misc.fromimage(Image.open(gts[i])).astype(float)/255.0

    psnr = output_psnr_mse(data, gt)
    psnrs.append(psnr)
print("PSNR:", np.mean(psnrs))

"""
75 pth
rp: 6 PSNR: 22.6392712102

"""
