import torch
import torch.nn as nn
import math
import numpy as np
import os
from os import listdir
from os.path import join
import torchvision.transforms as transforms

def weights_init_kaiming(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
  elif classname.find('Linear') != -1:
    nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
  elif classname.find('BatchNorm') != -1:
    # nn.init.uniform(m.weight.data, 1.0, 0.02)
    m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
    nn.init.constant(m.bias.data, 0.0)


def output_psnr_mse(img_orig, img_out):
  squared_error = np.square(img_orig - img_out)
  mse = np.mean(squared_error)
  psnr = 10 * np.log10(1.0 / mse)
  return psnr


def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])


def load_all_image(path):
  return [join(path, x) for x in listdir(path) if is_image_file(x)]


def save_checkpoint(model, epoch, model_folder):
  model_out_path = "checkpoints/%s/%d.pth" % (model_folder, epoch)

  state_dict = model.module.state_dict()
  for key in state_dict.keys():
    state_dict[key] = state_dict[key].cpu()

  if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

  if not os.path.exists("checkpoints/" + model_folder):
    os.makedirs("checkpoints/" + model_folder)

  torch.save({
    'epoch': epoch,
    'state_dict': state_dict}, model_out_path)
  print("Checkpoint saved to {}".format(model_out_path))


class FeatureExtractor(nn.Module):
  def __init__(self, cnn, feature_layer=11):
    super(FeatureExtractor, self).__init__()
    self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

  def forward(self, x):
    return self.features(x)


def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std