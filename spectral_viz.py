import numpy as np
import scipy.linalg
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
import pyvista as pv

def downsample_mesh(mesh, target_number=5000):
    """Downsample the mesh to a target number of vertices."""
    current_number = mesh.n_points
    if target_number >= current_number:
        return mesh  # No downsampling needed if target number is greater than current number
    target_reduction = 1 - (target_number / current_number)
    mesh_d = mesh.decimate(target_reduction)
    return mesh_d

def compute_affinity_matrix(points):
    """
    Compute the affinity matrix using a Gaussian kernel.
    """
    pairwise_dists = squareform(pdist(points, 'euclidean'))
    sigma = np.mean(pairwise_dists)  # Using the average of distances as sigma
    return np.exp(-pairwise_dists ** 2 / (2. * sigma ** 2))

def perform_spectral_clustering(points, n_clusters=17):
    """
    Perform spectral clustering on the given points.
    """
    # Step 1: Compute the affinity matrix
    W = compute_affinity_matrix(points)

    # Step 2: Normalize W to N
    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.linalg.inv(scipy.linalg.sqrtm(D))
    N = np.dot(np.dot(D_inv_sqrt, W), D_inv_sqrt)

    # Step 3: Compute k largest eigenvectors of N
    eigvals, eigvecs = scipy.linalg.eigh(N)
    V = eigvecs[:, -n_clusters:]  # Taking the last k eigenvectors

    # Step 4: Normalize rows of V to unit length
    V_normalized = V / np.linalg.norm(V, axis=1, keepdims=True)

    # Step 5 and 6: K-means clustering on rows of V_normalized
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    labels = kmeans.fit_predict(V_normalized)

    return labels

def visualize_with_pyvista(mesh, labels):
    """
    Visualize the mesh with labels using PyVista with a discrete color scale.
    """
    mesh['labels'] = labels
    # Create a discrete colormap
    n_clusters = np.unique(labels).size
    cmap = plt.cm.get_cmap('tab20', n_clusters)  # Using a colormap with sufficient distinct colors
    colors = cmap(np.linspace(0, 1, n_clusters))  # Generate colors
    # Convert colors to a format acceptable by PyVista
    colormap = mcolors.ListedColormap(colors)
    # Plotting
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='labels', cmap=colormap, clim=[0, n_clusters - 1])
    plotter.show()

def main(obj_file, target_points=1000):
    # Load the mesh
    mesh = pv.read(obj_file)
    # Downsample the mesh
    downsampled_mesh = downsample_mesh(mesh, target_points)
    # Perform spectral clustering
    labels = perform_spectral_clustering(np.array(downsampled_mesh.points), n_clusters=15)
    # Visualize the result
    visualize_with_pyvista(downsampled_mesh, labels)

# Example usage
#main('ZOUIF2W4_upper.obj')
