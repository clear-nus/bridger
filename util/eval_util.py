# import torch
# import torch.nn.functional as F
#
#
# def gaussian_kernel(x, y, sigma=1.0):
#     """
#     Compute the Gaussian kernel matrix between two sets of samples x and y.
#
#     Args:
#     - x (Tensor): Set of samples.
#     - y (Tensor): Set of samples.
#     - sigma (float): Bandwidth parameter for the Gaussian kernel.
#
#     Returns:
#     - Tensor: Gaussian kernel matrix.
#     """
#     xx = torch.sum(x * x, dim=1, keepdim=True)
#     yy = torch.sum(y * y, dim=1, keepdim=True)
#     xy = torch.mm(x, y.t())
#
#     return torch.exp(-0.5 * (xx - 2 * xy + yy) / (sigma**2))
#
#
# def mmd(x, y, sigma=1.0):
#     """
#     Compute the Maximum Mean Discrepancy (MMD) between two sets of samples x and y.
#
#     Args:
#     - x (Tensor): Set of samples.
#     - y (Tensor): Set of samples.
#     - sigma (float): Bandwidth parameter for the Gaussian kernel.
#
#     Returns:
#     - Tensor: MMD value.
#     """
#     x_kernel = gaussian_kernel(x, x, sigma)
#     y_kernel = gaussian_kernel(y, y, sigma)
#     xy_kernel = gaussian_kernel(x, y, sigma)
#
#     mmd_value = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
#     return mmd_value.item()

import torch
import torch.nn as nn


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


if __name__ == "__main__":
    # Example usage:
    # Generate two sets of samples from normal distributions
    sample1 = torch.randn((100, 2))
    sample2 = torch.randn((100, 2))

    mmd_loss = MMD_loss()
    mmd_value = mmd_loss(sample1, sample2)
    # Calculate the MMD between the two sets of samples
    # mmd_value = mmd(sample1, sample2)

    print("Maximum Mean Discrepancy:", mmd_value.item())