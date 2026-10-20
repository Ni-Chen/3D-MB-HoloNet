import torch
from torch import nn
import torch.nn.functional as F


def fftshift2d(tensor):
    return torch.fft.fftshift(tensor, dim=(-2, -1))


def ifftshift2d(tensor):
    return torch.fft.ifftshift(tensor, dim=(-2, -1))


def ft2d(tensor):
    return ifftshift2d(torch.fft.fft2(fftshift2d(tensor), dim=(-2, -1)))


def ift2d(tensor):
    return ifftshift2d(torch.fft.ifft2(fftshift2d(tensor), dim=(-2, -1)))


class NonNegativeScalar(nn.Module):
    def __init__(self, reg_param):
        super().__init__()
        self.value = nn.Parameter(torch.empty(1, 1, 1))
        nn.init.xavier_normal_(self.value)
        self.reg_param = reg_param

    def expanded(self, shape):
        return self.value.expand(shape)


class ResidualBlock(nn.Module):
    def __init__(self, channels, reg_param):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.reg_param = reg_param
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, inputs):
        output = F.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return F.relu(output + inputs)


class Phase(nn.Module):
    def __init__(self, channels, reg_param):
        super().__init__()
        self.mu = NonNegativeScalar(reg_param)
        self.threshold = NonNegativeScalar(reg_param)
        self.forward_block = ResidualBlock(channels, reg_param)
        self.proximal_block = ResidualBlock(channels, reg_param)
        self.symmetry_block = ResidualBlock(channels, reg_param)
        self.reg_param = reg_param

    def forward(self, volume, hologram, otf):
        mu = self.mu.expanded(volume.shape[1:])
        backprojection = backward_wave_prop(hologram, otf)
        numerator = ft2d((mu * backprojection + volume).to(torch.complex64))
        denominator = 1 + mu * otf.abs().square()
        next_volume = ift2d(numerator / denominator.to(torch.complex64)).real

        forward_volume = self.forward_block(next_volume)
        threshold = self.threshold.expanded(next_volume.shape[1:])
        thresholded = torch.sign(forward_volume) * F.relu(
            torch.abs(forward_volume) - threshold
        )
        output = self.proximal_block(thresholded)
        symmetry = self.symmetry_block(forward_volume) - next_volume
        return output, symmetry

    def regularization_loss(self):
        loss = self.reg_param * (
            self.mu.value.square().sum() + self.threshold.value.square().sum()
        )
        for block in (self.forward_block, self.proximal_block, self.symmetry_block):
            loss = loss + self.reg_param * (
                block.conv1.weight.square().sum() + block.conv2.weight.square().sum()
            )
        return loss


def backward_wave_prop(hologram, otf):
    if otf.ndim == 3:
        otf = otf.unsqueeze(0)
    if otf.shape[0] == 1 and hologram.shape[0] != 1:
        otf = otf.expand(hologram.shape[0], -1, -1, -1)
    otf = otf.to(torch.complex64)
    hologram = hologram.to(torch.complex64).unsqueeze(1)
    field = torch.fft.ifft2(
        torch.fft.fft2(hologram, dim=(-2, -1)) * otf.conj(), dim=(-2, -1)
    )
    return field.real


class MOHoloNet(nn.Module):
    def __init__(
        self,
        img_rows=64,
        img_cols=64,
        img_depths=5,
        layer_num=9,
        loss_sym_param=2e-3,
        reg_param=1e-5,
    ):
        super().__init__()
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_depths = img_depths
        self.loss_sym_param = loss_sym_param
        self.reg_param = reg_param
        self.phases = nn.ModuleList(
            Phase(img_depths, reg_param) for _ in range(layer_num)
        )
        self.output_bn = nn.BatchNorm2d(img_depths, eps=1e-3, momentum=0.01)

    def forward(self, hologram, otf, return_aux=False):
        if hologram.ndim != 3:
            raise ValueError("hologram must have shape [batch, height, width]")
        if hologram.shape[-2:] != (self.img_rows, self.img_cols):
            raise ValueError("hologram dimensions do not match the model")
        if otf.ndim == 3:
            otf = otf.unsqueeze(0)
        if otf.shape[1:] != (self.img_depths, self.img_rows, self.img_cols):
            raise ValueError("otf must have shape [batch, depth, height, width]")
        volume = backward_wave_prop(hologram, otf)
        symmetry_losses = []
        for phase in self.phases:
            volume, symmetry = phase(volume, hologram, otf)
            symmetry_losses.append(symmetry.square().mean())
        output = torch.sigmoid(self.output_bn(volume))
        symmetry_loss = self.loss_sym_param * torch.stack(symmetry_losses).mean()
        if return_aux:
            return output, symmetry_loss
        return output

    def regularization_loss(self):
        losses = [phase.regularization_loss() for phase in self.phases]
        return torch.stack(losses).sum() if losses else self.output_bn.weight.new_zeros(())

    @torch.no_grad()
    def project_nonnegative_parameters(self):
        for phase in self.phases:
            phase.mu.value.clamp_(min=0)
            phase.threshold.value.clamp_(min=0)
