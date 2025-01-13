import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np
from physics.radon.filters import RampFilter
from physics.radon.utils import PI, SQRT2, deg2rad, affine_grid, grid_sample

'''source: https://github.com/matteo-ronchetti/torch-radon'''


class Radon2(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True, dtype=torch.float):
        super(Radon2, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):

        reco = torch.zeros((1, 1, 512, 3), device=x.device)
        sqrt2 = 2 ** 0.5

        # 优化循环操作
        # 对于第一个循环
        for i in range(256):
            reco[0, 0, i + 128, 0] += torch.sum(x[0, 0, :, i])
        # 45

        for i in range(256):
            for k in range(256 - i):
                reco[0, 0, i + 256, 1]+= x[0, 0, k, k + i] * sqrt2
        reco[0, 0,  256, 1] = 0
        for i in range(256):
            for k in range(256 - i):
                reco[0, 0, -i + 256, 1] += x[0, 0, k + i, k] * sqrt2

        # # 45
        #
        # for i in range(256):
        #     for k in range(256 - i):
        #         reco[0, 0, i + 256, 1] = reco[0, 0, i + 256, 1] + x[0, 0, k , k+i] *sqrt2
        # for i in range(256):
        #     for k in range(256 - i):
        #         if i == 0 and k == 0:
        #             reco[0, 0, i + 256, 1] = 0
        #         reco[0, 0, -i + 256, 1] = reco[0, 0, -i + 256, 1] + x[0, 0, k+i, k] * sqrt2

        # 135
        for i in range(256):
            for k in range(256 - i):
                reco[0, 0, i + 256, 2] += x[0, 0, k + i, 255-k] * sqrt2

        reco[0, 0, 256, 2] = 0
        for i in range(181):
            for k in range(256 - i):
                reco[0, 0, -i + 256, 2] +=  x[0, 0, 255-k-i, k ] * sqrt2

        # # 135
        # for i in range(181):
        #     for k in range(256 - i):
        #         reco[0, 0, i + 181, 0] = reco[0, 0, i + 181, 0] + x[0, 0, k + i, 255-k] * (2 ** 0.5)
        # for i in range(181):
        #     for k in range(256 - i):
        #         if i == 0 and k == 0:
        #             reco[0, 0, i + 181, 0] = 0
        #         reco[0, 0, -i + 181, 0] = reco[0, 0, -i + 181,0] + x[0, 0, 255-k-i, k ] * (2 ** 0.5)
        # # 90
        # for i in range(256):
        #     for j in range(256):
        #          reco[0, 0, i + 53, 2] = reco[0, 0, i + 53, 2] + x[0, 0, i, j]



        N, C, W, H = x.shape
        assert (W == H)

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, C, W, len(self.theta), device=x.device, dtype=self.dtype)

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            out[..., i] = rotated.sum(2)
        out[0, 0, :363, :3] = reco[0, 0, :363, :3]
        # for i in range(363):
        #     for j in range(3):
        #         out[0, 0, i, j] = reco[0, 0, i, j]
        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta)
            # print(theta)
            # R = torch.tensor([[
            #     [1, 0, 0],
            #     [-theta.sin(), theta.cos(), 0],
            # ]], dtype=self.dtype)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, torch.Size([1, 1, grid_size, grid_size])))
        return all_grids

class Radon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True, dtype=torch.float):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta

        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):

        N, C, W, H = x.shape
        assert (W == H)

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, C, W, len(self.theta), device=x.device, dtype=self.dtype)

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            out[..., i] = rotated.sum(2)

        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta)
            # print(theta)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            # R = torch.tensor([[
            #     [1, 0, 0],
            #     [-theta.sin(), theta.cos(), 0],
            # ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, torch.Size([1, 1, grid_size, grid_size])))
        return all_grids

class IRadon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True,
                 use_filter=RampFilter(), out_size=None, dtype=torch.float):
        super(IRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            self.all_grids = self._create_grids(self.theta, in_size, circle)
        self.filter = use_filter if use_filter is not None else lambda x: x

    def forward(self, x):

        it_size = x.shape[2]
        ch_size = x.shape[1]

        if self.in_size is None:
            self.in_size = int((it_size / SQRT2).floor()) if not self.circle else it_size
        # if None in [self.ygrid, self.xgrid, self.all_grids]:
        if self.ygrid is None or self.xgrid is None or self.all_grids is None :
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.all_grids = self._create_grids(self.theta, self.in_size, self.circle)

        # sinogram
        x = self.filter(x)


        reco = torch.zeros(x.shape[0], ch_size, it_size, it_size, device=x.device, dtype=self.dtype)
        for i_theta in range(len(self.theta)):
            reco += grid_sample(x, self.all_grids[i_theta].repeat(reco.shape[0], 1, 1, 1).to(x.device))

        #
        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1]))

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(x.shape[0], ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.

        reco = reco * PI.item() / (2 * len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size) // 2
            reco = F.pad(reco, (pad, pad, pad, pad))

        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2 * in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype)
        return torch.meshgrid(unitrange, unitrange)

    def _XYtoT(self, theta):
        T = self.xgrid * (deg2rad(theta)).cos() - self.ygrid * (deg2rad(theta)).sin()
        return T

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for i_theta in range(len(angles)):
            X = torch.ones(grid_size, dtype=self.dtype).view(-1, 1).repeat(1, grid_size) * i_theta * 2. / (
                        len(angles) - 1) - 1.
            Y = self._XYtoT(angles[i_theta])
            all_grids.append(torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1).unsqueeze(0))
        return all_grids


class IRadon2(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True,
                 use_filter=RampFilter(), out_size=None, dtype=torch.float):
        super(IRadon2, self).__init__()

        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            self.all_grids = self._create_grids(self.theta, in_size, circle)
        self.filter = use_filter if use_filter is not None else lambda x: x

    def forward(self, x):

        it_size = x.shape[2]
        ch_size = x.shape[1]


        if self.in_size is None:
            self.in_size = int((it_size / SQRT2).floor()) if not self.circle else it_size
        # if None in [self.ygrid, self.xgrid, self.all_grids]:
        if self.ygrid is None or self.xgrid is None or self.all_grids is None :
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.all_grids = self._create_grids(self.theta, self.in_size, self.circle)

        # sinogram
        x = self.filter(x)

        # #0
        # reco2 = np.zeros((256, 256))
        # for i in range(157):
        #     for j in range(157):
        #         reco2[j + 50, i+50]  = x[0, 0, i + 103, 0] / 157
        # #45
        # reco3 = np.zeros((256, 256))
        # for i in range(157):
        #     for k in range(157 - i):
        #         reco3[ k+50, k+i+50] = (x[0, 0,181+i, 1] /(2 ** 0.5))/(157-i)
        # for i in range(157):
        #     for k in range(157 - i):
        #         reco3[k+50+i , k+50] = (x[0,0, 181 -i, 1] / (2 ** 0.5)) / (157 - i)
        # #135
        # reco4 = np.zeros((256, 256))
        # for i in range(157):
        #     for k in range(157 - i):
        #         reco4[ k+i+50, 205-k] = (x[0,0,181+i,2] /(2 ** 0.5))/157-i)
        # for i in range(157):
        #     for k in range(157 - i):
        #         reco4[205-k-i , k+50] = (x[0,0, 181 -i,2] / (2 ** 0.5)) / (157 - i)

        # 0
        reco2 = np.zeros((256, 256))
        for i in range(256):
            for j in range(256):
                reco2[j , i ] = x[0, 0, i + 53, 0] /256
        # 45
        reco3 = np.zeros((256, 256))
        for i in range(181):
            for k in range(256 - i):
                reco3[k , k + i ] = (x[0, 0, 181 + i,1] / (2 ** 0.5)) / (256 - i)
        for i in range(181):
            for k in range(256 - i):
                reco3[k  + i, k ] = (x[0, 0, 181 - i, 1] / (2 ** 0.5)) / (256 - i)
        # #135
        # reco4 = np.zeros((256, 256))
        # for i in range(181):
        #     for k in range(256 - i):
        #         reco4[ k+i, 255-k] = (x[0,0,181+i,0] /(2 ** 0.5))/(256-i)
        # for i in range(181):
        #     for k in range(256 - i):
        #         reco4[255-k-i , k] = (x[0,0, 181 -i,0] / (2 ** 0.5)) / (256 - i)
        # 90
        reco4 = np.zeros((256, 256))
        for i in range(256):
            for j in range(256):
                reco4[i , j ] = x[0, 0, i + 53, 2] /256



        reco = torch.zeros(x.shape[0], ch_size, it_size, it_size, device=x.device, dtype=self.dtype)
        for i_theta in range(len(self.theta)):
            reco += grid_sample(x, self.all_grids[i_theta].repeat(reco.shape[0], 1, 1, 1).to(x.device))

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1]))

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(x.shape[0], ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.

        reco = reco * PI.item() / (2 * len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size) // 2
            reco = F.pad(reco, (pad, pad, pad, pad))



        for i in range(256):
            for j in range(256):
                reco[0, 0, i , j ] = (reco2 [i, j]+ reco3 [i, j]+ reco4 [i, j])*256/3
        for i in range(256):
            for j in range(256):
                if i < 50 or j < 50 or i > 205 or j > 205:
                    reco[0, 0, i, j] = 0

        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2 * in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype)
        return torch.meshgrid(unitrange, unitrange)

    def _XYtoT(self, theta):
        T = self.xgrid * (deg2rad(theta)).cos() - self.ygrid * (deg2rad(theta)).sin()
        return T

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for i_theta in range(len(angles)):
            X = torch.ones(grid_size, dtype=self.dtype).view(-1, 1).repeat(1, grid_size) * i_theta * 2. / (
                        len(angles) - 1) - 1.
            Y = self._XYtoT(angles[i_theta])
            all_grids.append(torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1).unsqueeze(0))
        return all_grids


if __name__ == '__main__':
    img_width = 2
    num_proj = 180
    device = 'cuda:0'
    radon = Radon(in_size=img_width, theta=torch.arange(num_proj), circle=False).to(device)
    iradon = IRadon(in_size=img_width, theta=torch.arange(num_proj), circle=False).to(device)

    img = torch.randn([1, 1, 2, 2]).to(device)
    sinogram = radon(img)
    b_img = iradon(sinogram)
