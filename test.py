#!/usr/bin/env python
import argparse
import torch
import utils
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Resize

from model.rpnet import Net

parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--rb", type=int, default=18, help="number of residual blocks")
parser.add_argument("--checkpoint", type=str, help="path to load model checkpoint")
parser.add_argument("--test", type=str, help="path to load test images")

opt = parser.parse_args()
print(opt)

net = Net(opt.rb)
net.load_state_dict(torch.load(opt.checkpoint)['state_dict'])
net.eval()
net = nn.DataParallel(net, device_ids=[0, 1, 2, 3]).cuda()

print(net)

images = utils.load_all_image(opt.test)

for im_path in tqdm(images):
    filename = im_path.split('/')[-1]
    print(filename)
    im = Image.open(im_path)
    h, w = im.size
    print(h, w)
    im = ToTensor()(im)
    im = Variable(im).view(1, -1, w, h)
    im = im.cuda()
    with torch.no_grad():
        im = net(im)
    im = torch.clamp(im, 0., 1.)
    im = im.cpu()
    im = im.data[0]
    im = ToPILImage()(im)
    im.save('output/%s' % filename)
