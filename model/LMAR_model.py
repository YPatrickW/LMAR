from model import net
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize

try:
    from resize_right import resize
except:
    from .resize_right import resize

try:
    from .interp_methods import *
except:
    from interp_methods import *

from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor

import tinycudann as tcnn
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchviz import make_dot


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def get_local_grid(img):
    local_grid = make_coord(img.shape[-2:], flatten=False).cuda()
    local_grid = local_grid.permute(2, 0, 1).unsqueeze(0)
    local_grid = local_grid.expand(img.shape[0], 2, *img.shape[-2:])

    return local_grid

def creat_coord(x):
    b = x.shape[0]
    coord = make_coord(x.shape[-2:], flatten=False)
    coord = coord.permute(2, 0, 1).contiguous().unsqueeze(0)
    coord = coord.expand(b, 2, *coord.shape[-2:])

    coord_ = coord.clone()
    coord_ = coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
    coord_ = coord_.permute(0, 2, 3, 1).contiguous()
    coord_ = coord_.view(b, -1, coord.size(1))
    return coord.cuda(), coord_.cuda()


def get_cell(img, local_grid):
    cell = torch.ones_like(local_grid)
    cell[:, 0] *= 2 / img.size(2)
    cell[:, 1] *= 2 / img.size(3)

    return cell


class TcnnFCBlock(tcnn.Network):
    def __init__(
            self, in_features, out_features,
            num_hidden_layers, hidden_features,
            activation: str = 'LeakyRelu', last_activation: str = 'None',
            seed=42):
        assert hidden_features in [16, 32, 64, 128], "hidden_features can only be 16, 32, 64, or 128."
        super().__init__(in_features, out_features, network_config={
            "otype": "FullyFusedMLP",  # Component type.
            "activation": activation,  # Activation of hidden layers.
            "output_activation": last_activation,  # Activation of the output layer.
            "n_neurons": hidden_features,  # Neurons in each hidden layer. # May only be 16, 32, 64, or 128.
            "n_hidden_layers": num_hidden_layers,  # Number of hidden layers.
        }, seed=seed)

    def forward(self, x: torch.Tensor):
        prefix = x.shape[:-1]
        return super().forward(x.flatten(0, -2)).unflatten(0, prefix)


class LMAR_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.resume_flag = args.resume["flag"]
        self.load_path = args.resume["checkpoint"]

        if self.resume_flag and self.load_path:
            self.model = net(args)
            checkpoint = torch.load(self.load_path)
            self.model.load_state_dict(checkpoint["state_dict"])
            for param in self.model.parameters():
                param.requires_grad_(False)

        self.in_channel = 3
        self.out_channel = 3
        self.kernel_size = 3
        self.imnet = TcnnFCBlock(7, self.in_channel * self.out_channel * self.kernel_size * self.kernel_size, 5,
                                 128).cuda()
        self.mid_nodes = {"hr_backbone.skip2": "bottom"}
        self.extractor_mid = create_feature_extractor(self.model, self.mid_nodes)
        self.modulation = nn.Conv2d(6, 3, 1, 1, 0)
        # self.projection = nn.Conv2d()

    def forward(self, x, down_size, up_size, test_flag=False):
        if test_flag:
            up_out, _ = self.inference(x, down_size, up_size)
            return up_out, _ 
        else:
            down_x, hr_feature, new_lr_feature, ori_lr_feature, residual, res = self.train_model(x, down_size, up_size)
            return down_x, hr_feature, new_lr_feature, ori_lr_feature, residual, res

    def train_model(self, x, down_size, up_size):
        # down_sizer = transforms.Resize(size=down_size,
        #                                interpolation=transforms.InterpolationMode.BILINEAR)
        # up_sizer = transforms.Resize(size=up_size,
        #                              interpolation=transforms.InterpolationMode.BILINEAR)

        b = x.shape[0]
        # down_x = down_sizer(x)
        down_x = resize(x, out_shape=down_size, antialiasing=False)
        # down_x = resize(x, out_shape=down_size, antialiasing=True)

        hr_feature = self.extractor_mid(x)["bottom"]
        # feature_sizer = transforms.Resize(size=(hr_feature.shape[2], hr_feature.shape[3]),
        #                                   interpolation=transforms.InterpolationMode.BILINEAR)

        hr_coord, hr_coord_ = self.creat_coord(x)
        lr_coord, _ = self.creat_coord(down_x)
        q_coord = F.grid_sample(lr_coord, hr_coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)
        q_coord = q_coord.view(b, -1, hr_coord.size(2) * hr_coord.size(3)).permute(0, 2, 1).contiguous()

        # test_coord = F.grid_sample(lr_coord, hr_coord.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)
        # test_rel_coord = hr_coord - test_coord
        # test_rel_coord = test_rel_coord.view(b, -1, 2)

        # test_rel_coord[:, :, 0] *= down_x.shape[-2]
        # test_rel_coord[:, :, 1] *= down_x.shape[-1]

        rel_coord = hr_coord_ - q_coord
        rel_coord[:, :, 0] *= down_x.shape[-2]
        rel_coord[:, :, 1] *= down_x.shape[-1]

        laplacian = x - resize(down_x, out_shape=up_size, antialiasing=False)
        # laplacian = x - resize(down_x, out_shape=up_size, antialiasing=True)

        laplacian = laplacian.reshape(b, laplacian.size(1), -1).permute(0, 2, 1).contiguous()

        # cell
        hr_grid = self.get_local_grid(x)
        hr_cell = self.get_cell(x, hr_grid)
        hr_cell_ = hr_cell.clone()
        hr_cell_ = hr_cell_.permute(0, 2, 3, 1).contiguous()
        rel_cell = hr_cell_.view(b, -1, hr_cell.size(1))
        rel_cell[:, :, 0] *= down_x.shape[-2]
        rel_cell[:, :, 1] *= down_x.shape[-1]

        inp = torch.cat([rel_coord.cuda(), rel_cell.cuda(), laplacian], dim=-1)
        local_weight = self.imnet(inp)
        local_weight = local_weight.type(torch.float32)
        local_weight = local_weight.view(b, -1, x.shape[1] * 9, 3).contiguous()

        unfolded_x = F.unfold(x, 3, padding=1).view(b, -1, x.shape[2] * x.shape[3]).permute(0, 2, 1).contiguous()
        cols = unfolded_x.unsqueeze(2)
        out = torch.matmul(cols, local_weight).squeeze(2).permute(0, 2, 1).contiguous().view(b, -1, x.size(2),
                                                                                             x.size(3))
        out = resize(out, out_shape=down_size, antialiasing=False)
        # out = resize(out, out_shape=down_size, antialiasing=True)

        # out = down_sizer(out)

        # ori
        ori_lr_feature = self.extractor_mid(down_x)["bottom"]
        ori_lr_feature = resize(ori_lr_feature, out_shape=(hr_feature.shape[2], hr_feature.shape[3]),
                                antialiasing=False)
        # ori_lr_feature = resize(ori_lr_feature, out_shape=(hr_feature.shape[2], hr_feature.shape[3]), antialiasing=True)
        # ori_lr_feature = feature_sizer(ori_lr_feature)

        # new
        down_x = self.modulation(torch.cat([down_x, out], dim=1))
        new_lr_feature = self.extractor_mid(down_x)["bottom"]

        new_lr_feature = resize(new_lr_feature, out_shape=(hr_feature.shape[2], hr_feature.shape[3]),
                                antialiasing=False)
        # new_lr_feature = resize(new_lr_feature, out_shape=(hr_feature.shape[2], hr_feature.shape[3]), antialiasing=True)

        # new_lr_feature = feature_sizer(new_lr_feature)

        # res = resize(self.model(self.modulation(torch.cat([down_x, out], dim=1))), out_shape=up_size,
        #              antialiasing=False)

        # res = up_sizer(self.model(self.modulation(torch.cat([down_x, out], dim=1))))
        res = 0

        return down_x, hr_feature, \
               new_lr_feature, ori_lr_feature, out, res

    def inference(self, x, down_size, up_size):
        b = x.shape[0]
        down_x = resize(x, out_shape=down_size, antialiasing=False)
        hr_coord, hr_coord_ = self.creat_coord(x)
        lr_coord, _ = self.creat_coord(down_x)
        q_coord = F.grid_sample(lr_coord, hr_coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)
        q_coord = q_coord.view(b, -1, hr_coord.size(2) * hr_coord.size(3)).permute(0, 2, 1).contiguous()

        rel_coord = hr_coord_ - q_coord
        rel_coord[:, :, 0] *= down_x.shape[-2]
        rel_coord[:, :, 1] *= down_x.shape[-1]

        hr_grid = self.get_local_grid(x)
        hr_cell = self.get_cell(x, hr_grid)

        hr_cell_ = hr_cell.clone()
        hr_cell_ = hr_cell_.permute(0, 2, 3, 1).contiguous()

        rel_cell = hr_cell_.view(b, -1, hr_cell.size(1))
        rel_cell[:, :, 0] *= down_x.shape[-2]
        rel_cell[:, :, 1] *= down_x.shape[-1]

        laplacian = x - resize(down_x, out_shape=up_size, antialiasing=False)
        # laplacian = x - resize(down_x, out_shape=up_size, antialiasing=True)

        laplacian = laplacian.reshape(b, laplacian.size(1), -1).permute(0, 2, 1).contiguous()
        # laplacian = F.unfold(laplacian, 3, padding=1).view(b, -1, laplacian.shape[2] * laplacian.shape[3]).permute(0, 2, 1).contiguous()

        inp = torch.cat([rel_coord.cuda(), rel_cell.cuda(), laplacian], dim=-1)
        local_weight = self.imnet(inp)
        local_weight = local_weight.type(torch.float32)
        local_weight = local_weight.view(b, -1, x.shape[1] * 9, 3)

        unfolded_x = F.unfold(x, 3, padding=1).view(b, -1, x.shape[2] * x.shape[3]).permute(0, 2, 1).contiguous()

        cols = unfolded_x.unsqueeze(2)

        out = torch.matmul(cols, local_weight).squeeze(2).permute(0, 2, 1).contiguous().view(b, -1, x.size(2),
                                                                                             x.size(3))
        out = resize(out, out_shape=down_size, antialiasing=False)
        down_x = self.modulation(torch.cat([down_x, out], dim=1))

        res = resize(self.model(down_x), out_shape=up_size, antialiasing=False)
        return res, down_x

    def creat_coord(self, x):
        b = x.shape[0]
        coord = make_coord(x.shape[-2:], flatten=False)
        coord = coord.permute(2, 0, 1).contiguous().unsqueeze(0)
        coord = coord.expand(b, 2, *coord.shape[-2:])

        coord_ = coord.clone()
        coord_ = coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        coord_ = coord_.permute(0, 2, 3, 1).contiguous()
        coord_ = coord_.view(b, -1, coord.size(1))
        return coord.cuda(), coord_.cuda()

    def get_local_grid(self, img):
        local_grid = make_coord(img.shape[-2:], flatten=False).cuda()
        local_grid = local_grid.permute(2, 0, 1).unsqueeze(0)
        local_grid = local_grid.expand(img.shape[0], 2, *img.shape[-2:])

        return local_grid

    def get_cell(self, img, local_grid):
        cell = torch.ones_like(local_grid)
        cell[:, 0] *= 2 / img.size(2)
        cell[:, 1] *= 2 / img.size(3)

        return cell

