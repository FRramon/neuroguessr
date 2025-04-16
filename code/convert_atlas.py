import nibabel as nib
import numpy as np
import os

# Load the probabilistic atlas
prob_atlas = nib.load('/Users/francoisramon/Downloads/HippoAmyg/HippoAmygProbs.MNIsymSpace.left.nii.gz')
prob_data = prob_atlas.get_fdata()

# Assuming each volume (along the 4th dimension) represents a different subfield
num_regions = prob_data.shape[3]

# Create a new 3D volume for the discrete atlas
# Each voxel will contain the index (1 to N) of the most probable region
single_volume = np.argmax(prob_data, axis=3) + 1  # +1 so regions start at 1 instead of 0

# Set voxels with all zeros across regions to background (0)
sum_prob = np.sum(prob_data, axis=3)
single_volume[sum_prob == 0] = 0

# Create a new NIfTI image with the same header/affine as the original
discrete_atlas = nib.Nifti1Image(single_volume, prob_atlas.affine, prob_atlas.header)

# Save the new discrete atlas
nib.save(discrete_atlas, "/Users/francoisramon/Downloads/Brainstem/Brainstem-thr0.nii.gz")