import sys
import numpy as np
import nibabel as nib
from skimage import measure
import trimesh
import pyvista as pv
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifti_to_mesh_pyvista.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def nifti_to_obj(nifti_path, output_obj_path, threshold=1, smooth_iterations=2):
    logging.info(f"Converting NIfTI {nifti_path} to OBJ {output_obj_path}")
    try:
        # Load NIfTI file
        img = nib.load(nifti_path)
        data = img.get_fdata().astype(np.int32)
        logging.debug(f"NIfTI data shape: {data.shape}, unique values: {np.unique(data)}")
        
        # Threshold: set values >= threshold to 1, others to 0
        binary_data = (data >= threshold).astype(np.uint8)
        logging.debug(f"After thresholding (>= {threshold}): {np.sum(binary_data)} voxels are 1")
        
        # Generate mesh using marching cubes
        verts, faces, _, _ = measure.marching_cubes(binary_data, level=0.5)
        logging.debug(f"Mesh generated: {verts.shape[0]} vertices, {faces.shape[0]} faces")
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Smooth the mesh using Laplacian smoothing
        for _ in range(smooth_iterations):
            mesh = trimesh.smoothing.filter_laplacian(mesh)
        logging.debug("Mesh smoothed")
        
        # Clean the mesh
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.fix_normals()
        logging.debug(f"Mesh cleaned: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
        
        # Apply affine transformation to vertices
        affine = img.affine
        verts_transformed = trimesh.transform_points(mesh.vertices, affine)
        mesh.vertices = verts_transformed
        
        # Export to OBJ
        mesh.export(output_obj_path)
        logging.info(f"OBJ file saved to {output_obj_path}")
        return mesh, affine
    except Exception as e:
        logging.error(f"Error in nifti_to_obj: {str(e)}")
        raise

def main():
    logging.info("Starting main function")
    try:
        # Convert NIfTI to OBJ
        input_nifti = "/Users/francoisramon/Desktop/These/neuroguessr/data/xtract.nii.gz"
        output_obj = "/Users/francoisramon/Desktop/These/neuroguessr/data/atlas_mesh.obj"
        crosshair_coords = [47, 77, 114]  # Example NIfTI coordinates (x, y, z)
        mesh, affine = nifti_to_obj(input_nifti, output_obj)
        
        # Convert trimesh to pyvista mesh
        faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).ravel()
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        
        # Transform crosshair coordinates to world space
        crosshair_coords = np.array([crosshair_coords], dtype=np.float32)
        world_coords = trimesh.transform_points(crosshair_coords, affine)[0]
        logging.debug(f"Crosshair world coordinates: {world_coords}")
        
        # Create crosshair as three lines
        crosshair_size = 5.0
        lines = [
            # X-axis
            pv.Line(
                [world_coords[0] - crosshair_size, world_coords[1], world_coords[2]],
                [world_coords[0] + crosshair_size, world_coords[1], world_coords[2]]
            ),
            # Y-axis
            pv.Line(
                [world_coords[0], world_coords[1] - crosshair_size, world_coords[2]],
                [world_coords[0], world_coords[1] + crosshair_size, world_coords[2]]
            ),
            # Z-axis
            pv.Line(
                [world_coords[0], world_coords[1], world_coords[2] - crosshair_size],
                [world_coords[0], world_coords[1], world_coords[2] + crosshair_size]
            )
        ]
        
        # Create plotter and display
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color='blue')
        for line in lines:
            plotter.add_mesh(line, color='red', line_width=2)
        plotter.show()
        logging.info("PyVista plot displayed")
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()