import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from torch.nn.functional import sigmoid
from monai.inferers import sliding_window_inference

# ---------------- Trainable Frangi Layers ----------------

class TrainableGaussian3D(nn.Module):
    def __init__(self, channels=1, size=7):
        super().__init__()
        coords = torch.arange(size) - size // 2
        grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing='ij')
        kernel = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2 * 1.0**2))
        kernel = kernel / kernel.sum()
        self.kernel = nn.Parameter(kernel.view(1,1,size,size,size).repeat(channels,1,1,1,1))

    def forward(self, x):
        padding = self.kernel.shape[-1] // 2
        return nn.functional.conv3d(x, self.kernel, padding=padding, groups=x.shape[1])

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
        C = H - q[...,None,None]*I
        p = torch.sqrt((C**2).sum((-2,-1)) / 6)
        B = C / (p[...,None,None] + 1e-10)
        detB = torch.linalg.det(B)
        theta = torch.acos(torch.clamp(detB/2, -1, 1)) / 3
        eigvals = [2*torch.cos(theta+2*np.pi*k/3)*p + q for k in range(3)]
        return torch.sort(torch.stack(eigvals, dim=-1), dim=-1).values

class TrainableFrangiVesselnessLayer(nn.Module):
    def __init__(self, init_beta=0.4, init_c=15.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))
        self.c = nn.Parameter(torch.tensor(init_c))

    def forward(self, l1,l2,l3):
        cond = (l2<0) & (l3<0)
        Ra = torch.abs(l2/(l3+1e-10))
        S = torch.sqrt(l1**2 + l2**2 + l3**2)
        vesselness = torch.zeros_like(l1)
        vesselness[cond] = (1 - torch.exp(-Ra[cond]**2/(2*self.beta**2))) * \
                           torch.exp(-S[cond]**2/(2*self.c**2))
        return vesselness

class TrainableFrangiFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gaussian = TrainableGaussian3D()
        self.hessian = Hessian3D()
        self.eig = Eigenvalues3x3()
        self.frangi = TrainableFrangiVesselnessLayer()

    def forward(self,x):
        smoothed = self.gaussian(x)
        H = self.hessian(smoothed)
        eigvals = self.eig(H)
        l1,l2,l3 = eigvals[...,0], eigvals[...,1], eigvals[...,2]
        return self.frangi(l1,l2,l3)

# ---------------- U-Net Components ----------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch,out_ch,3,padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch,out_ch,3,padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self,in_ch=2,out_ch=1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch,32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(32,64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(64,128)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(128,256)
        self.up3 = nn.ConvTranspose3d(256,128,2,stride=2)
        self.dec3 = ConvBlock(256,128)
        self.up2 = nn.ConvTranspose3d(128,64,2,stride=2)
        self.dec2 = ConvBlock(128,64)
        self.up1 = nn.ConvTranspose3d(64,32,2,stride=2)
        self.dec1 = ConvBlock(64,32)
        self.out_conv = nn.Conv3d(32,out_ch,1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        if d3.shape[-3:] != e3.shape[-3:]:
            d3 = nn.functional.interpolate(d3, size=e3.shape[-3:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3,e3],dim=1))
        d2 = self.up2(d3)
        if d2.shape[-3:] != e2.shape[-3:]:
            d2 = nn.functional.interpolate(d2, size=e2.shape[-3:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2,e2],dim=1))
        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]:
            d1 = nn.functional.interpolate(d1, size=e1.shape[-3:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1,e1],dim=1))
        return self.out_conv(d1)

class TrainableFrangiNetUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.frangi = TrainableFrangiFeatureExtractor()
        self.unet = UNet3D(in_ch=2,out_ch=1)

    def forward(self,x):
        ct_only = x[:,0:1,...]
        vesselness = self.frangi(ct_only)
        return self.unet(torch.cat([ct_only, vesselness], dim=1))

# ---------------- Helper Functions ----------------

def load_volume(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine, nii.header

def normalize_ct(volume):
    volume = np.clip(volume, -1000, 400)
    return (volume - volume.min()) / (volume.max() - volume.min() + 1e-10)

def safe_inference(model, x_tensor, roi_size=(128,128,128), sw_batch_size=2):
    return sliding_window_inference(
        x_tensor,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=0.25
    )

def predict_and_save(model_path, folder, output_folder, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrainableFrangiNetUNet3D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_folder, exist_ok=True)
    cases = sorted([f for f in os.listdir(folder) if f.endswith('_ct.nii')])

    for ct_file in cases:
        name = ct_file.replace('_ct.nii', '')
        print(f"🔍 Predicting: {name}")

        # Load & normalize
        ct, aff, hdr = load_volume(os.path.join(folder, f"{name}_ct.nii"))
        ct = normalize_ct(ct)
        x_tensor = torch.tensor(ct[None,None,...], dtype=torch.float32).to(device)

        # Sliding-window inference
        with torch.no_grad():
            pred_logits = safe_inference(model, x_tensor, roi_size=(128,128,128), sw_batch_size=2)
            pred_prob = sigmoid(pred_logits)[0,0].cpu().numpy()

        # Save probability map
        prob_nii = nib.Nifti1Image(pred_prob.astype(np.float32), affine=aff, header=hdr)
        nib.save(prob_nii, os.path.join(output_folder, f"{name}_prob.nii.gz"))

        # Save thresholded mask
        mask = (pred_prob > threshold).astype(np.uint8)
        mask_nii = nib.Nifti1Image(mask, affine=aff, header=hdr)
        nib.save(mask_nii, os.path.join(output_folder, f"{name}_pred.nii.gz"))

        print(f"✅ Saved: {name}_prob.nii.gz & {name}_pred.nii.gz")

if __name__ == "__main__":
    predict_and_save(
        model_path="epoch90.pth",
        folder="/rsrch9/ip/fkhalaj/Desktop/whole/FinalCTs",
        output_folder="/rsrch9/ip/fkhalaj/Desktop/whole/FinalCTs/Result",
        threshold=0.5
    )
