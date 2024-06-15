import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import yaml
import torchvision.transforms as transforms
from utils import read_args, save_checkpoint, AverageMeter, CosineAnnealingWarmRestarts
import time
from tqdm import trange, tqdm
from torchvision.utils import save_image
import json
import time
import logging
import torch
from torch import nn, optim
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn as nn

from data import *
from model import *
from loss import *
import pyiqa

from torch.autograd import Variable

global_step = 0
psnr_calculator = pyiqa.create_metric('psnr').cuda()
ssim_calculator = pyiqa.create_metric('ssimc', downsample=True).cuda()

feature_extractor = VGGPerceptualLoss(resize=False).cuda()
feature_extractor.eval()


def weight_annealing(epoch):
    initial_weight = 1
    if epoch < 2:
        return initial_weight  # 初始阶段保持权重不变
    else:
        return initial_weight * 0.001  # 后续阶段权重继续减小


def train(model, data_loader, criterion, optimizer, epoch, args):
    global global_step
    iter_bar = tqdm(data_loader, desc='Iter (loss=X.XXX)')
    nbatches = len(data_loader)

    total_losses = AverageMeter()

    pixel_losses = AverageMeter()
    resize_losses = AverageMeter()
    pseudo_losses = AverageMeter()

    psnrs = AverageMeter()
    ssims = AverageMeter()

    optimizer.zero_grad()

    start_time = time.time()

    if not os.path.exists(args.output_dir + '/image_train'):
        os.mkdir(args.output_dir + '/image_train')

    if not os.path.exists(args.output_dir + "/models"):
        os.mkdir(args.output_dir + "/models")

    for i, batch in enumerate(iter_bar):
        optimizer.zero_grad()
        inp_img, gt_img, down_h, down_w, inp_img_path = batch
        batch_size = inp_img.size(0)
        inp_img = inp_img.cuda()
        gt_img = gt_img.cuda()

        down_size = (down_h.item(), down_w.item())
        up_size = eval(args.train_loader["img_size"])


        down_x, hr_feature, new_lr_feature, ori_lr_feature, residual, res = model(inp_img, down_size, up_size)

        pixel_loss = criterion(new_lr_feature, hr_feature)

        pixel_losses.update(pixel_loss.item(), batch_size)
        
        resize_loss = feature_extractor(down_x, resize(inp_img, out_shape=down_size, antialiasing=False),
                                        feature_layers=[3])
        resize_loss = resize_loss * weight_annealing(epoch)
        resize_losses.update(resize_loss.item(), batch_size)


        pseudo_loss, _ = feat_ssim(new_lr_feature, hr_feature, inp_img)
        pseudo_losses.update(pseudo_loss.item(), batch_size)

        total_loss = pixel_loss + resize_loss + pseudo_loss
        total_losses.update(total_loss.item(), batch_size)

        total_loss.backward()

        optimizer.step()

        iter_bar.set_description('Iter (loss=%5.6f)' % total_losses.avg)

        if i % 200 == 0:
            # print(residual.max())
            error = torch.abs(resize(inp_img, out_shape=down_size, antialiasing=False) - down_x)
            # error = (error - error.min()) / (error.max()-error.min())
            saved_image = torch.cat(
                [resize(inp_img, out_shape=down_size, antialiasing=False)[0:2], down_x[0:2], error[0:2]],
                dim=0)
            save_image(saved_image, args.output_dir + '/image_train/epoch_{}_iter_down_{}.png'.format(epoch, i))

            saved_image = torch.cat(
                [torch.mean(hr_feature, dim=1, keepdim=True)[0:2], torch.mean(new_lr_feature, dim=1, keepdim=True)[0:2],
                 torch.mean(ori_lr_feature, dim=1, keepdim=True)[0:2],
                 torch.mean(torch.abs(new_lr_feature - ori_lr_feature), dim=1, keepdim=True)[0:2]],
                dim=0)
            save_image(saved_image, args.output_dir + '/image_train/epoch_{}_iter_feat_{}.png'.format(epoch, i))
            # residual = (residual - residual.min()) / (residual.max()-residual.min())
            residual = residual * 10
            save_image(residual[0], args.output_dir + '/image_train/epoch_{}_iter_out_{}.png'.format(epoch, i))

        if i % max(1, nbatches // 10) == 0:
            psnr_val, ssim_val = 0.0, 0.0
            psnrs.update(psnr_val, batch_size)
            ssims.update(ssim_val, batch_size)

            logging.info(
                "Epoch {}, learning rates {:}, Iter {}, total_loss {:.4f}, pixel_loss {:.4f}, resize_loss {:.4f}, pseudo_loss {:.4f}, PSNR {:.4f}, SSIM {:.4f}, Elapse time {:.2f}\n".format(
                    epoch, optimizer.param_groups[0]["lr"], i, total_losses.avg, pixel_losses.avg, resize_losses.avg,
                    pseudo_losses.avg,
                    psnrs.avg, ssims.avg,
                    time.time() - start_time))

    if epoch % 1 == 0:
        logging.info("** ** * Saving model and optimizer ** ** * ")

        output_model_file = os.path.join(args.output_dir + "/models", "model.%d.bin" % (epoch))
        state = {"epoch": epoch, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "step": global_step}

        save_checkpoint(state, output_model_file)
        logging.info("Save model to %s", output_model_file)

    logging.info(
        "Finish training epoch %d, avg total_loss: %.4f, avg pixel_loss: %.4f, avg resize_loss: %.4f, avg pseudo_loss: %.4f, avg PSNR: %.2f, avg SSIM: %.2F, and takes %.2f seconds" % (
            epoch, total_losses.avg, pixel_losses.avg, resize_losses.avg, pseudo_losses.avg, psnrs.avg, ssims.avg,
            time.time() - start_time))

    logging.info("***** CUDA.empty_cache() *****\n")
    torch.cuda.empty_cache()


def evaluate(model, load_path, data_loader, epoch):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()

    psnrs = AverageMeter()
    ssims = AverageMeter()
    random_index = torch.randint(low=0, high=5, size=(1,))
    down_size = eval(args.test_loader["img_size"])
    down_size = down_size[random_index]
    logging.info("Inference at down size: {}".format(down_size))
    up_size = eval(args.test_loader["gt_size"])

    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            inp_img, gt_img, inp_img_path = batch
            inp_img = inp_img.cuda()
            batch_size = inp_img.size(0)
            up_out = model(inp_img, down_size, up_size, test_flag=True)

            # metrics
            clamped_out = torch.clamp(up_out, 0, 1)
            psnr_val, ssim_val = psnr_calculator(clamped_out, gt_img), ssim_calculator(clamped_out, gt_img)
            psnrs.update(torch.mean(psnr_val).item(), batch_size)
            ssims.update(torch.mean(ssim_val).item(), batch_size)
            torch.cuda.empty_cache()

            if i % 100 == 0:
                logging.info(
                    "PSNR {:.4f}, SSIM {:.4f}, Elapse time {:.2f}\n".format(psnrs.avg, ssims.avg,
                                                                            time.time() - start_time))

        logging.info(f"Finish test at epoch {epoch}: avg PSNR: %.4f, avg SSIM: %.4F, and takes %.2f seconds" % (
            psnrs.avg, ssims.avg, time.time() - start_time))


def main(args):
    global global_step

    start_epoch = 1
    global_step = 0

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=2)

    log_format = "%(asctime)s %(levelname)-8s %(message)s"
    log_file = os.path.join(args.output_dir, "train_log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    logging.info(args.__dict__)

    model = LMAR_model(args)

    optimizer = optim.Adam(model.parameters(), lr=args.optimizer["lr"],
                           betas=(0.9, 0.999))

    logging.info("Building data loader")

    if args.train_loader["loader"] == "resize":
        train_transforms = transforms.Compose([transforms.Resize(eval(args.train_loader["img_size"])),
                                               transforms.ToTensor()])
        train_loader = get_loader(args.data["train_dir"],
                                  eval(args.train_loader["img_size"]), train_transforms, False,
                                  int(args.train_loader["batch_size"]), args.train_loader["num_workers"],
                                  args.train_loader["shuffle"], random_flag=False)

    elif args.train_loader["loader"] == "crop":
        train_loader = get_loader(args.data["train_dir"],
                                  eval(args.train_loader["img_size"]), False, True,
                                  int(args.train_loader["batch_size"]), args.train_loader["num_workers"],
                                  args.train_loader["shuffle"], random_flag=args.train_loader["random_flag"])

    elif args.train_loader["loader"] == "default":
        train_transforms = transforms.Compose([transforms.ToTensor()])
        train_loader = get_loader(args.data["train_dir"],
                                  eval(args.train_loader["img_size"]), train_transforms, False,
                                  int(args.train_loader["batch_size"]), args.train_loader["num_workers"],
                                  args.train_loader["shuffle"], random_flag=args.train_loader["random_flag"])
    else:
        raise NotImplementedError

    if args.test_loader["loader"] == "default":

        test_transforms = transforms.Compose([transforms.ToTensor()])
        test_loader = get_loader(args.data["test_dir"],
                                 eval(args.test_loader["img_size"]), test_transforms, False,
                                 int(args.test_loader["batch_size"]), args.test_loader["num_workers"],
                                 args.test_loader["shuffle"], random_flag=False)

    elif args.test_loader["loader"] == "resize":
        test_transforms = transforms.Compose([transforms.Resize(eval(args.test_loader["img_size"])),
                                              transforms.ToTensor()])
        test_loader = get_loader(args.data["test_dir"],
                                 eval(args.test_loader["img_size"]), test_transforms, False,
                                 int(args.test_loader["batch_size"]), args.test_loader["num_workers"],
                                 args.test_loader["shuffle"], random_flag=False)
    else:
        raise NotImplementedError

    criterion = nn.SmoothL1Loss()
    # vgg_loss = VGGLoss()

    if args.optimizer["type"] == "cos":
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.optimizer["T_0"],
                                                   T_mult=args.optimizer["T_MULT"],
                                                   eta_min=args.optimizer["ETA_MIN"],
                                                   last_epoch=-1)
    elif args.optimizer["type"] == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.optimizer["step"],
                                                       gamma=args.optimizer["gamma"])

    t_total = int(len(train_loader) * args.optimizer["total_epoch"])
    logging.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    logging.info("***** Running training *****")
    logging.info("  Batch size = %d", args.train_loader["batch_size"])
    logging.info("  Num steps = %d", t_total)
    logging.info("  Loader length = %d", len(train_loader))

    model.train()
    model.cuda()

    logging.info("Begin training from epoch = %d\n", start_epoch)
    # evaluate(model, "/home/yuwei/experiment/cvpr/prompt_final_vgg_gradient/models/model.1.bin", test_loader, 1)
    for epoch in trange(start_epoch, args.optimizer["total_epoch"] + 1, desc="Epoch"):
        train(model, train_loader, criterion, optimizer, epoch, args)
        lr_scheduler.step()
        if epoch % args.evaluate_intervel == 0:
            logging.info("***** Running testing *****")
            load_path = os.path.join(args.output_dir + "/models", "model.%d.bin" % (epoch))
            evaluate(model, load_path, test_loader, epoch)
            logging.info("***** End testing *****")


if __name__ == '__main__':
    parser = read_args("/home/yuwei/code/cvpr/config/LMAR_config.yaml")
    args = parser.parse_args()
    main(args)
