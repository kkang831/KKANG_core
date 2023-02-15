#!/usr/bin/env python
import os, sys, argparse, glob
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt', default='', help='')
    return parser.parse_args()

def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location='cpu')

    print('---ckpt.keys()---')
    print(ckpt.keys())

    print(ckpt['state_dict'].keys())
#     for key, value in ckpt.items():
#         print(key)
#         inspect(value)        

# def inspect(item):
#     if type(item) is dict:
#         print('!!!')
#         for key, value in item.items():
#             print(key)
#             inspect(value)
#     elif type(item) is list:
#         for i in range(len(item)):
#             inspect(item[i])
#     else:
#         try:
#             print(item.shape)
#         except:
#             print(type(item))

if __name__ == '__main__':
    main()

# //TODO: 구현해야함