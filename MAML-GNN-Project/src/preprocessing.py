import os
import pydicom
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib

# Ensure that the results directory exists
os.makedirs('results', exist_ok=True)

def load_dicom_series(dicom_path):
    """Load a series of DICOM files or a single DICOM file and return a 3D numpy array."""
    if os.path.isdir(dicom_path):
        dicom_files = [pydicom.dcmread(os.path.join(dicom_path, f)) for f in sorted(os.listdir(dicom_path))]
        dicom_files.sort(key=lambda x: int(x.InstanceNumber))  # Sort based on the instance number
        slices = [file.pixel_array for file in dicom_files]
    else:
        # Handle the case where dicom_path is a single DICOM file
        dicom_files = [pydicom.dcmread(dicom_path)]
        slices = [dicom_files[0].pixel_array]
        
    volume = np.stack(slices, axis=-1)
    return volume

def save_nifti_image(data, output_path):
    """Save the 3D numpy array as a NIfTI image."""
    new_img = nib.Nifti1Image(data, affine=np.eye(4))  # Assuming identity matrix for affine
    nib.save(new_img, output_path)

def crop_and_resize(volume, target_shape=(200, 200, 33)):
    """Crop the region of interest and resize to the target shape."""
    center = np.array(volume.shape) // 2
    start = center - np.array(target_shape) // 2
    end = start + np.array(target_shape)
    start = np.maximum(start, 0)
    end = np.minimum(end, volume.shape)
    cropped_volume = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    zoom_factors = np.array(target_shape) / np.array(cropped_volume.shape)
    resized_volume = zoom(cropped_volume, zoom_factors, order=1)
    return resized_volume

def label_nodule(malignancy_score):
    """Label the nodule as benign or malignant based on the malignancy score."""
    if malignancy_score < 3:
        return 0  # Benign
    else:
        return 1  # Malignant

def preprocess_and_label(dicom_path, malignancy_score, output_path):
    volume = load_dicom_series(dicom_path)
    resized_volume = crop_and_resize(volume)
    label = label_nodule(malignancy_score)
    save_nifti_image(resized_volume, output_path)
    return label

# Example paths (adjust to your actual file paths in Google Colab)
lidc_ct_path = "/content/1-001.dcm"  # This could be a single DICOM file
lung_ct_path = "/content/1-042.dcm"  # This is a directory containing DICOM files
pet_path = "/content/1-056.dcm"  # This is a directory containing DICOM files

# Ensure output directories exist
os.makedirs('results', exist_ok=True)

# Example malignancy scores
lidc_malignancy_score = 2  # Example value; replace with actual score
lung_ct_malignancy_score = 4  # Example value; replace with actual score
pet_malignancy_score = 5  # Example value; replace with actual score

# Output paths
lidc_ct_output_path = "results/output_lidc_ct_image.nii.gz"
lung_ct_output_path = "results/output_lung_ct_image.nii.gz"
pet_output_path = "results/output_pet_image.nii.gz"

# Preprocess the images and label the nodules
lidc_label = preprocess_and_label(lidc_ct_path, lidc_malignancy_score, lidc_ct_output_path)
lung_ct_label = preprocess_and_label(lung_ct_path, lung_ct_malignancy_score, lung_ct_output_path)
pet_label = preprocess_and_label(pet_path, pet_malignancy_score, pet_output_path)

print(f"LIDC-CT Label: {lidc_label} (0=Benign, 1=Malignant)")
print(f"LUNG-CT Label: {lung_ct_label} (0=Benign, 1=Malignant)")
print(f"PET Label: {pet_label} (0=Benign, 1=Malignant)")
