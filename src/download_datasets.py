import os
import torch
import argparse
from torchvision.datasets import CIFAR10, CIFAR100

import coloredlogs, logging
coloredlogs.install()

# https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
def dir_path(string):
  if os.path.isdir(string):
    return string
  else:
    raise NotADirectoryError(string)


if __name__ == "__main__":
    # Parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--path', type=dir_path, required=True, 
    help='Download cifar10 and cifar100 to this path.')

  args = parser.parse_args()
  c10_path = args.path + '/cifar10'
  c100_path = args.path + '/cifar100'

  if not os.path.exists(c10_path):
    os.makedirs(c10_path)

  if not os.path.exists(c100_path):
    os.makedirs(c100_path)

  CIFAR10(c10_path, download=True)
  CIFAR100(c100_path, download=True)
