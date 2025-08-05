import os
import torch
import nibabel as nib
import numpy as np
def normalize_image(image):
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
    return image
from torch.utils.data import Dataset, DataLoader

class VesselSegmentationDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cases = []
        for f in os.listdir(data_dir):
            if f.endswith("_ct.nii"):
                case = f.split("_")[0]
                label_path = os.path.join(data_dir, f"{case}_label.nii")
                input_path = os.path.join(data_dir, f"{case}_ct.nii")
                if os.path.exists(label_path) and os.path.exists(input_path):
                    self.cases.append(case)
        self.cases = sorted(self.cases)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        input_path = os.path.join(self.data_dir, f"{case}_ct.nii")
        label_path = os.path.join(self.data_dir, f"{case}_label.nii")

        input_img = nib.load(input_path).get_fdata().astype(np.float32)
        label_img = nib.load(label_path).get_fdata().astype(np.float32)

        
        input_img = np.expand_dims(input_img, axis=0)
        label_img = np.expand_dims(label_img, axis=0)

        
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)

        input_tensor = torch.tensor(input_img, dtype=torch.float32)
        label_tensor = torch.tensor(label_img, dtype=torch.float32)

        return input_tensor, label_tensor


if __name__ == "__main__":
    data_dir = r"/rsrch9/ip/fkhalaj/Desktop/whole/First_training"  
    dataset = VesselSegmentationDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_cases = 0

    for i, (image, label) in enumerate(dataloader):
        print(f" Batch {i}")
        print("  Image shape:", image.shape)
        print("  Label shape:", label.shape)
        total_cases += 1

    print(f"\n Total cases processed: {total_cases}")
