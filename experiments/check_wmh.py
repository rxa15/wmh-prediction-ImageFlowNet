import nibabel as nib
import numpy as np
import os

wmh_dir = "/app/dataset/LBC1936/Scan1_Wave2_WMH"  # ganti ke folder WMH kamu

for i, fname in enumerate(os.listdir(wmh_dir)):
    if not fname.endswith(".nii") and not fname.endswith(".nii.gz"):
        continue

    path = os.path.join(wmh_dir, fname)
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)

    unique_vals = np.unique(data)
    print(f"{fname}: min={data.min()}, max={data.max()}, uniques={unique_vals[:5]}")
    
    if i >= 9:  # cek 10 file pertama
        break
