import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

folder = '/rsrch9/ip/fkhalaj/Desktop/whole/'

for filename in os.listdir(folder):
    if filename.endswith('_pred.nii'):
        base_name = filename.replace('_pred.nii', '')

        pred_path = os.path.join(folder, f"{base_name}_pred.nii")
        mask_path = os.path.join(folder, f"{base_name}_mask.nii")
        output_path = os.path.join(folder, f"{base_name}_pred_overlap_only.nii")

        if not os.path.exists(mask_path):
            print(f"❌ Skipping {base_name}: mask file not found.")
            continue
        
        pred_img = nib.load(pred_path)
        mask_img = nib.load(mask_path)
        pred_data = pred_img.get_fdata()
        mask_data = mask_img.get_fdata()

        print(f"\n=== Processing {base_name} ===")
        print(f"Prediction shape: {pred_data.shape}, Mask shape: {mask_data.shape}")
        print(f"Prediction max: {np.max(pred_data)}")
        print(f"Mask labels (unique): {np.unique(mask_data)}")
        
        if pred_data.shape != mask_data.shape:
            zoom_factors = np.array(pred_data.shape) / np.array(mask_data.shape)
            mask_data = zoom(mask_data, zoom_factors, order=0)
            print("🔁 Resampled mask to match prediction.")
    
        overlap_mask = np.isin(mask_data, [1, 2])  
        overlap_mask = overlap_mask.astype(pred_data.dtype)
    
        masked_pred = pred_data * overlap_mask
        
        print(f"✅ Max after masking: {np.max(masked_pred)}")
        print(f"✅ Voxels kept (overlap): {np.sum(masked_pred > 0)}")
        
        output_img = nib.Nifti1Image(masked_pred, affine=pred_img.affine, header=pred_img.header)
        nib.save(output_img, output_path)
        print(f"💾 Saved: {output_path}")
