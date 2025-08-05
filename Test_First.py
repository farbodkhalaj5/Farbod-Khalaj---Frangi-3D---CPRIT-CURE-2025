import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from glob import glob


class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU()
            )
        self.enc1 = CBR(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = CBR(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = CBR(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = CBR(128, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = CBR(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = CBR(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = CBR(64, 32)
        self.out_conv = nn.Conv3d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frangi = nn.Identity()  # placeholder
        self.unet = UNet3D()

    def forward(self, x):
        return self.unet(x)

def resize_to_target(volume, target_shape=(256, 256, 256)):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def run_folder_inference(pth_path, input_folder, label_folder, output_folder, threshold=0.5):
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FullModel().to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device), strict=False)
    model.eval()

    input_files = sorted(glob(os.path.join(input_folder, "*.nii*")))

    for input_file in input_files:
        case_name = os.path.basename(input_file).replace("_input.nii", "").replace("_input.nii.gz", "")
        label_file = os.path.join(label_folder, f"{case_name}_label.nii.gz")
        if not os.path.exists(label_file):
            label_file = os.path.join(label_folder, f"{case_name}_label.nii")
            if not os.path.exists(label_file):
                print(f"❌ Label not found for {case_name}, skipping.")
                continue

        input_vol = nib.load(input_file)
        input_data = input_vol.get_fdata().astype(np.float32)
        label_data = nib.load(label_file).get_fdata().astype(np.float32)

        input_resized = resize_to_target(input_data)
        label_resized = resize_to_target(label_data)
        input_resized = (input_resized - input_resized.min()) / (input_resized.max() - input_resized.min() + 1e-8)
        input_tensor = torch.tensor(input_resized).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

        pred_binary = (pred > threshold).astype(np.uint8)
        seg_nii = nib.Nifti1Image(pred_binary, affine=input_vol.affine)
        seg_path = os.path.join(output_folder, f"{case_name}_segmentation.nii.gz")
        nib.save(seg_nii, seg_path)

        z = pred.shape[2] // 2
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(input_resized[:, :, z], cmap='gray'); axs[0].set_title("Input")
        axs[1].imshow(label_resized[:, :, z], cmap='gray'); axs[1].set_title("Ground Truth")
        axs[2].imshow(pred_binary[:, :, z], cmap='gray'); axs[2].set_title("Predicted")
        for ax in axs: ax.axis("off")
        preview_path = os.path.join(output_folder, f"{case_name}_preview.png")
        plt.savefig(preview_path)
        plt.close()

        print(f"✅ {case_name} done. Seg: {seg_path}, Preview: {preview_path}")


run_folder_inference(
    pth_path=r"/rsrch9/ip/fkhalaj/Desktop/whole/First_training/epoch90.pth",
    input_folder=r"/rsrch9/ip/fkhalaj/Desktop/whole/First_training/input",
    label_folder=r"/rsrch9/ip/fkhalaj/Desktop/whole/First_training/label",
    output_folder=r"/rsrch9/ip/fkhalaj/Desktop/whole/First_training/Prediction1"
)
