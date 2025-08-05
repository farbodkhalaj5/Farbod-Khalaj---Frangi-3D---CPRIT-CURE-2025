import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import torchio as tio
from torch.utils.data import random_split
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings("ignore", message="Using TorchIO images without a torchio.SubjectsLoader")


def pad_if_needed(volume, target_size):
    c, d, h, w = volume.shape
    pad_d = max(0, target_size - d)
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(
            volume,
            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant"
        )
    return volume


def random_crop_3d(volume, label, size=128):
    volume = pad_if_needed(volume, size)
    label = pad_if_needed(label, size)
    z, y, x = volume.shape[1:]
    dz = random.randint(0, max(0, z - size))
    dy = random.randint(0, max(0, y - size))
    dx = random.randint(0, max(0, x - size))
    vol_crop = volume[:, dz:dz+size, dy:dy+size, dx:dx+size]
    lbl_crop = label[:, dz:dz+size, dy:dy+size, dx:dx+size]
    return vol_crop, lbl_crop


class PatchCTDataset(torch.utils.data.Dataset):
    def __init__(self, folder, patch_size=128, patches_per_case=16):
        self.folder = folder
        self.patch_size = patch_size
        self.patches_per_case = patches_per_case
        self.inputs = sorted([f for f in os.listdir(folder) if f.endswith('_ct.nii')])

        self.aug = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            tio.RandomNoise(std=0.05),
            tio.RandomAffine(scales=0.1, degrees=10)
        ])

    def __len__(self):
        return len(self.inputs) * self.patches_per_case

    def __getitem__(self, idx):
        case_idx = idx // self.patches_per_case
        name = self.inputs[case_idx].replace('_ct.nii', '')
        ct = nib.load(os.path.join(self.folder, f"{name}_ct.nii")).get_fdata()
        label = nib.load(os.path.join(self.folder, f"{name}_label.nii")).get_fdata()
        ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-10)
        x = ct[None, ...]
        y = label[None, ...]
        x_crop, y_crop = random_crop_3d(x, y, self.patch_size)
        subject = tio.Subject(
            x=tio.ScalarImage(tensor=x_crop),
            y=tio.LabelMap(tensor=y_crop)
        )
        return self.aug(subject)


class TrainableGaussian3D(nn.Module):
    def __init__(self, channels=1, size=7):
        super().__init__()
        coords = torch.arange(size) - size // 2
        grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing='ij')
        kernel = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2 * 1.0**2))
        kernel = kernel / kernel.sum()
        self.kernel = nn.Parameter(kernel.view(1, 1, size, size, size).repeat(channels, 1, 1, 1, 1))

    def forward(self, x):
        padding = self.kernel.shape[-1] // 2
        return F.conv3d(x, self.kernel, padding=padding, groups=x.shape[1])

class Hessian3D(nn.Module):
    def forward(self, x):
        grads = torch.gradient(x, dim=(2,3,4))
        Hxx = torch.gradient(grads[0], dim=2)[0]
        Hyy = torch.gradient(grads[1], dim=3)[0]
        Hzz = torch.gradient(grads[2], dim=4)[0]
        Hxy = torch.gradient(grads[0], dim=3)[0]
        Hxz = torch.gradient(grads[0], dim=4)[0]
        Hyz = torch.gradient(grads[1], dim=4)[0]
        H = torch.stack([
            torch.stack([Hxx, Hxy, Hxz], dim=-1),
            torch.stack([Hxy, Hyy, Hyz], dim=-1),
            torch.stack([Hxz, Hyz, Hzz], dim=-1)
        ], dim=-2)
        return H

class Eigenvalues3x3(nn.Module):
    def forward(self, H):
        q = H.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) / 3
        I = torch.eye(3, device=H.device).expand(H.shape)
        C = H - q[..., None, None] * I
        p = torch.sqrt((C**2).sum((-2, -1)) / 6)
        B = C / (p[..., None, None] + 1e-10)
        detB = torch.linalg.det(B)
        theta = torch.acos(torch.clamp(detB / 2, -1, 1)) / 3
        eigvals = [2 * torch.cos(theta + 2 * np.pi * k / 3) * p + q for k in range(3)]
        return torch.sort(torch.stack(eigvals, dim=-1), dim=-1).values

class TrainableFrangiVesselnessLayer(nn.Module):
    def __init__(self, init_alpha=0.5, init_beta=0.5, init_c=15.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))
        self.c = nn.Parameter(torch.tensor(init_c))

    def forward(self, l1, l2, l3):
        cond = (l2 < 0) & (l3 < 0)
        RA = torch.abs(l2 / (l3 + 1e-10))
        RB = torch.abs(l1 / torch.sqrt(torch.abs(l2 * l3) + 1e-10))
        S = torch.sqrt(l1**2 + l2**2 + l3**2)
        vesselness = torch.zeros_like(l1)
        vesselness[cond] = (1 - torch.exp(-RA[cond]**2 / (2 * self.alpha**2))) * \
                           torch.exp(-RB[cond]**2 / (2 * self.beta**2)) * \
                           (1 - torch.exp(-S[cond]**2 / (2 * self.c**2)))
        return vesselness

class TrainableFrangiFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gaussian = TrainableGaussian3D()
        self.hessian = Hessian3D()
        self.eig = Eigenvalues3x3()
        self.frangi = TrainableFrangiVesselnessLayer()

    def forward(self, x):
        smoothed = self.gaussian(x)
        H = self.hessian(smoothed)
        eigvals = self.eig(H)
        l1, l2, l3 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]
        return self.frangi(l1, l2, l3)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): 
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=2, out_ch=1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(128, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out_conv = nn.Conv3d(32, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        if d3.shape[-3:] != e3.shape[-3:]:
            d3 = F.interpolate(d3, size=e3.shape[-3:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-3:] != e2.shape[-3:]:
            d2 = F.interpolate(d2, size=e2.shape[-3:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]:
            d1 = F.interpolate(d1, size=e1.shape[-3:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)

class TrainableFrangiNetUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.frangi = TrainableFrangiFeatureExtractor()
        self.unet = UNet3D(in_ch=2, out_ch=1)

    def forward(self, x):
        vesselness = self.frangi(x)
        return self.unet(torch.cat([x, vesselness], dim=1))


def dice_loss(pred, target, smooth=1e-5):
    pred_sig = torch.sigmoid(pred)
    intersection = (pred_sig * target).sum()
    return 1 - (2.*intersection + smooth)/(pred_sig.sum() + target.sum() + smooth)

def combo_loss(pred, target):
    pred_sig = torch.sigmoid(pred)
    return 0.5 * dice_loss(pred, target) + 0.5 * F.binary_cross_entropy(pred_sig, target)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    for batch in loader:
        x = batch['x'][tio.DATA].cuda()
        y = batch['y'][tio.DATA].cuda()
        out = model(x)
        loss = combo_loss(out, y)
        total_loss += loss.item()
    return total_loss / len(loader)


def train(folder, epochs=50, batch_size=1, lr=1e-3):
    dataset = PatchCTDataset(folder, patch_size=128)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = tio.SubjectsLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = tio.SubjectsLoader(val_dataset, batch_size=1, shuffle=False)
    model = TrainableFrangiNetUNet3D().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = batch['x'][tio.DATA].cuda()
            y = batch['y'][tio.DATA].cuda()
            out = model(x)
            loss = combo_loss(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader)
        print(f"✅ Epoch {epoch} | TrainLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(folder, f"frangi_unet_epoch{epoch}.pth"))
        scheduler.step()
    return model

if __name__ == "__main__":
    train(folder="/rsrch9/ip/fkhalaj/Desktop/whole", epochs=10000)
