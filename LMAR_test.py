import argparse
import yaml
import torchvision.transforms as transforms
from utils import read_args, save_checkpoint, AverageMeter, CosineAnnealingWarmRestarts
import time
from tqdm import trange, tqdm
from torchvision.utils import save_image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import time
import logging
import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F

import copy
from model import *
from data import *
from PIL import Image
from torch.optim import LBFGS
import pyiqa
from thop import profile
from thop import clever_format

from torchvision.models.feature_extraction import create_feature_extractor

psnr_calculator = pyiqa.create_metric('psnr').cuda()
ssim_calculator = pyiqa.create_metric('ssimc', downsample=True).cuda()


def test(load_path, data_loader, args):
    model = LMAR_model(args)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()

    psnrs = AverageMeter()
    ssims = AverageMeter()
    lpipss = AverageMeter()
    niqes = AverageMeter()
    
    down_size = (1440, 2560)
    logging.info("Inference at down size: {}".format(down_size))
    up_size = eval(args.test_loader["gt_size"])

    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            inp_img, gt_img, inp_img_path = batch
            inp_img = inp_img.cuda()
            batch_size = inp_img.size(0)
            gt_img = gt_img.cuda()
            up_out = model(inp_img, down_size, up_size, test_flag=True)
            name = inp_img_path[0].split("/")[-1]
            # save_image(up_out[0], os.path.join(save_path, name))
    
            # metrics
            clamped_out = torch.clamp(up_out, 0, 1)

            psnr_val, ssim_val = psnr_calculator(clamped_out, gt_img), ssim_calculator(clamped_out, gt_img)
            psnrs.update(psnr_val.item(), batch_size)
            ssims.update(ssim_val.item(), batch_size)
    
            if i % 700 == 0:
                logging.info(
                    "PSNR {:.4f}, SSIM {:.4f}, LPIPS {:.4F}, NIQE {:.4F}, Elapse time {:.2f}\n".format(psnrs.avg, ssims.avg, lpipss.avg, niqes.avg,
                                                                            time.time() - start_time))
    
        logging.info("Finish test: avg PSNR: %.4f, avg SSIM: %.4F, avg LPIPS: %.4F, avg NIQE: %.4F, and takes %.2f seconds" % (
            psnrs.avg, ssims.avg, lpipss.avg, niqes.avg, time.time() - start_time))


def main(args, load_path):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    test_transforms = transforms.Compose([transforms.ToTensor()])

    log_format = "%(asctime)s %(levelname)-8s %(message)s"
    log_file = os.path.join(args.output_dir, "test_log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info("Building data loader")

    test_loader = get_loader(args.data["test_dir"],
                             eval(args.test_loader["img_size"]), test_transforms, False,
                             int(args.test_loader["batch_size"]), args.test_loader["num_workers"],
                             args.test_loader["shuffle"], random_flag=False)
    test_time(load_path, test_loader, args)


if __name__ == '__main__':
    parser = read_args("./config/LMAR_config.yaml")
    args = parser.parse_args()
    main(args, "./pretrained_models\LMAR_model.bin")
