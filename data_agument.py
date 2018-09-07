import os
import numpy as np
from PIL import Image
import argparse
from scipy.misc import imsave
from scipy.ndimage import rotate
from joblib import Parallel, delayed

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument("--size", type=int, default=512, help="which size to generate")
parser.add_argument('--fold_A', dest='fold_A', help='input directory for Haze Image', type=str,
                    default='../dataset/IndoorTrainHazy')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for Clear Image', type=str,
                    default='../dataset/IndoorTrainGT')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/IndoorTrain')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

fix_size = int(args.size)
splits = os.listdir(args.fold_A)
folder = args.fold_AB

if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs("%s/label" % folder)
    os.makedirs("%s/data" % folder)


def arguments(sp):
    print("Process %s" % sp)
    count_im = 0
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, '_'.join([sp.split('_')[0], sp.split('_')[1], 'GT' + '.' + sp.split('_')[-1].split('.')[-1]]))

    for flip in [0, 1, 2]:
        for degree in [0, 1, 2, 3]:

            im_A = np.asarray(Image.open(img_fold_A))
            im_B = np.asarray(Image.open(img_fold_B))

            if flip == 1:
                im_A = np.flip(im_A, 0)
                im_B = np.flip(im_B, 0)
            if flip == 2:
                im_A = np.flip(im_A, 1)
                im_B = np.flip(im_B, 1)

            if degree != 0:
                im_A = rotate(im_A, 90 * degree)
                im_B = rotate(im_B, 90 * degree)

            h, w, c = im_A.shape

            for x in range(0, h, fix_size // 2):
                for y in range(0, w, fix_size // 2):

                    if x + fix_size < h and y + fix_size < w:
                        patch_A = im_A[x:x + fix_size, y:y + fix_size]
                        patch_B = im_B[x:x + fix_size, y:y + fix_size]

                        imsave("%s/data/%d_%s.png" % (folder, count_im, '_'.join(sp.split('_')[:-1])), patch_A)
                        imsave("%s/label/%d_%s.png" % (folder, count_im, '_'.join(sp.split('_')[:-1])), patch_B)
                        count_im += 1
    print("Process %s for %d" % (sp, count_im))


Parallel(-1)(delayed(arguments)(sp) for sp in splits)