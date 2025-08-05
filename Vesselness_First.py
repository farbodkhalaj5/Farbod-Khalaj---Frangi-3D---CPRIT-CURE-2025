import os
import numpy as np
import nibabel as nib
from scipy.ndimage import (
    gaussian_filter, binary_closing, generate_binary_structure,
    binary_dilation, binary_fill_holes, label
)

def compute_hessian_smoothed(volume, sigma=1):
    smoothed = gaussian_filter(volume, sigma=sigma)
    gradients = np.gradient(smoothed)
    hessian = np.empty((*volume.shape, 3, 3))
    for i in range(3):
        for j in range(3):
            hessian[..., i, j] = np.gradient(gradients[i], axis=j)
    return hessian

def robust_eigenvalues_3x3(H):
    q = np.trace(H, axis1=-2, axis2=-1) / 3
    I = np.eye(3)
    C = H - q[..., None, None] * I
    p = np.sqrt(np.sum(C**2, axis=(-2, -1)) / 6)
    B = C / (p[..., None, None] + 1e-10)
    detB = np.linalg.det(B)
    theta = np.arccos(np.clip(detB / 2, -1, 1)) / 3
    eigvals = [2 * np.cos(theta + 2 * np.pi * k / 3) * p + q for k in range(3)]
    return np.sort(np.stack(eigvals, axis=-1), axis=-1)

def frangi_vesselness_3d(volume, sigmas=None, beta=0.4, c=15):
    if sigmas is None:
        sigmas = np.arange(1, 4.1, 0.1)
    vesselness = np.zeros_like(volume)
    for sigma in sigmas:
        H = compute_hessian_smoothed(volume, sigma)
        eigvals = robust_eigenvalues_3x3(H)
        l1, l2, l3 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]
        cond = (l2 < 0) & (l3 < 0)
        Ra = np.abs(l2 / (l3 + 1e-10))
        S = np.sqrt(l1**2 + l2**2 + l3**2)
        response = np.zeros_like(l1)
        response[cond] = (1 - np.exp(-Ra[cond]**2 / (2 * beta**2))) * np.exp(-S[cond]**2 / (2 * c**2))
        vesselness = np.maximum(vesselness, response)
    return vesselness

def restrict_range(volume, low=130, high=200):
    return (volume >= low) & (volume <= high)

def postprocess_mask(mask):
    binary = mask > 0.1
    struct = generate_binary_structure(3, 2)
    closed = binary_closing(binary, structure=struct, iterations=2)
    dilated = binary_dilation(closed, structure=struct, iterations=1)
    filled = binary_fill_holes(dilated)
    labeled, num = label(filled)
    if num == 0:
        return np.zeros_like(filled, dtype=np.float32)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = np.argmax(sizes)
    return (labeled == largest_label).astype(np.float32)

def process_volume(volume_np):
    mask = restrict_range(volume_np, 130, 200)
    masked_input = (volume_np * mask).astype(np.float32)
    if masked_input.max() > 0:
        masked_input = (masked_input - masked_input.min()) / (masked_input.max() - masked_input.min() + 1e-10)
    vesselness = frangi_vesselness_3d(masked_input)
    vesselness_masked = vesselness * mask
    cleaned = postprocess_mask(vesselness_masked)
    return cleaned

def main():
    input_folder = r"/rsrch9/ip/fkhalaj/Desktop/whole/First_training"
    output_folder = r"/rsrch9/ip/fkhalaj/Desktop/whole/First_training/Vesselness"
    os.makedirs(output_folder, exist_ok=True)

    all_inputs = sorted([f for f in os.listdir(input_folder) if f.endswith("_ct.nii")])

    if not all_inputs:
        print(" No input files found.")
        return

    for i, filename in enumerate(all_inputs, 1):
        input_path = os.path.join(input_folder, filename)
        output_name = filename.replace("_ct.nii", "_label.nii")
        output_path = os.path.join(output_folder, output_name)

        print(f" [{i}/{len(all_inputs)}] Processing: {filename}")
        try:
            nifti = nib.load(input_path)
            ct_volume = nifti.get_fdata().astype(np.float32)
            vessel_output = process_volume(ct_volume)
            nib.save(nib.Nifti1Image(vessel_output, nifti.affine), output_path)
            print(f" Saved: {output_name}")
        except Exception as e:
            print(f" Failed on {filename}: {e}")
            
if __name__ == "__main__":
    main()
