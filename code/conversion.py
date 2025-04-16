import nibabel as nib
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import colorsys
import os

# Load the atlas NIfTI file
wdir = '/Users/francoisramon/Desktop/These/neuroguessr/data'
atlas_img = nib.load(os.path.join(wdir, "Cerebellum-MNIfnirt-maxprob-thr25-1mm.nii.gz"))
atlas_data = atlas_img.get_fdata().astype(int)

# Find unique labels in the atlas (voxel values)
voxel_values = np.unique(atlas_data)

# Parse the XML file to get label indices and names
tree = ET.parse(os.path.join(wdir, "Cerebellum_MNIfnirt.xml"))
root = tree.getroot()
label_dict = {}
for label in root.findall('.//label'):
    index = int(label.get('index'))
    name = label.text.strip().replace(' ', '-')  # Replace spaces with hyphens
    label_dict[index] = name

# Map NIfTI voxel values to XML indices (voxel value = XML index + 1)
# Create a dictionary to map voxel values to names
voxel_to_name = {0: 'Background'}  # Voxel value 0 is background
for xml_index in label_dict:
    voxel_value = xml_index + 1
    voxel_to_name[voxel_value] = label_dict[xml_index]

# Print mapping for verification
print("Voxel value to region mapping:")
for v in sorted(voxel_to_name.keys()):
    print(f"Voxel value {v}: {voxel_to_name[v]}")

# Build the adjacency graph (6-connected neighbors) using voxel values
shape = atlas_data.shape
offsets = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
adjacency = defaultdict(set)
for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            label = atlas_data[x, y, z]
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                    nlabel = atlas_data[nx, ny, nz]
                    if nlabel != label:
                        adjacency[label].add(nlabel)
                        adjacency[nlabel].add(label)

# Perform greedy graph coloring
colors = {}
for label in sorted(voxel_values):
    if label == 0:
        continue  # Skip background
    used_colors = set(colors.get(neighbor, 0) for neighbor in adjacency[label])
    color = 1  # Start colors from 1
    while color in used_colors:
        color += 1
    colors[label] = color

# Determine the number of colors used
max_color = max(colors.values()) if colors else 0

# Generate distinct RGB colors
def get_distinct_colors(n):
    if n == 0:
        return []
    hsv_tuples = [(x * 1.0 / n, 0.7, 0.9) for x in range(n)]  # Increase saturation and value for brighter colors
    rgb_tuples = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_tuples]
    return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb_tuples]

color_list = get_distinct_colors(max_color)

# Write the output LUT file with the specified header
with open(os.path.join(wdir, "Cerebellum_MNIfnirt.txt"), 'w') as f:
    f.write('#No. Label Name:                            R   G   B   A\n')
    for label in sorted(voxel_values):
        name = voxel_to_name.get(label, f'Label_{label}')
        if label == 0:
            r, g, b = 0, 0, 0  # Background gets black
        elif label in colors:
            color_index = colors[label] - 1  # Adjust for 0-based indexing
            r, g, b = color_list[color_index]
        else:
            r, g, b = 0, 0, 0  # Default to black if not colored
        # Format the line with spaces for alignment
        f.write(f'{label:<4} {name:<35} {r:<3} {g:<3} {b:<3} 0\n')
