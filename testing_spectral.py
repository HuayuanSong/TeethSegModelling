import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import scipy.linalg
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import pyvista as pv

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['labels'])

def downsample_mesh(mesh, target_number=5000):
    current_number = mesh.n_points
    if target_number >= current_number:
        return mesh
    target_reduction = 1 - (target_number / current_number)
    mesh_d = mesh.decimate(target_reduction)
    return mesh_d

def compute_affinity_matrix(points):
    pairwise_dists = squareform(pdist(points, 'euclidean'))
    sigma = np.mean(pairwise_dists)
    return np.exp(-pairwise_dists ** 2 / (2. * sigma ** 2))

def perform_spectral_clustering(points, n_clusters=17):
    W = compute_affinity_matrix(points)
    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.linalg.inv(scipy.linalg.sqrtm(D))
    N = np.dot(np.dot(D_inv_sqrt, W), D_inv_sqrt)
    eigvals, eigvecs = scipy.linalg.eigh(N)
    V = eigvecs[:, -n_clusters:]
    V_normalized = V / np.linalg.norm(V, axis=1, keepdims=True)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    labels = kmeans.fit_predict(V_normalized)
    return labels

def your_prediction_function(obj_file, n_clusters=17, target_points=1000):
    mesh = pv.read(obj_file)
    downsampled_mesh = downsample_mesh(mesh, target_points)
    labels = perform_spectral_clustering(np.array(downsampled_mesh.points), n_clusters)
    return labels

def calculate_miou(predicted_labels, true_labels, num_classes):
    matrix = confusion_matrix(true_labels, predicted_labels, labels=np.arange(num_classes))
    miou = np.mean([matrix[i, i] / (np.sum(matrix[i, :]) + np.sum(matrix[:, i]) - matrix[i, i])
                    for i in range(num_classes)])
    return miou

def main(csv_file):
    df = pd.read_csv(csv_file)
    num_classes = 17  # Update based on your data

    miou_scores = []
    for _, row in df.iterrows():
        obj_path = row['obj_file_path']
        json_path = row['json_file_path']

        mesh = pv.read(obj_path)
        # Downsample the mesh
        downsampled_mesh = downsample_mesh(mesh, 5000)
        # Perform spectral clustering
        predicted_labels = perform_spectral_clustering(np.array(downsampled_mesh.points), 15)

        true_labels = read_json(json_path)
        miou = calculate_miou(predicted_labels, true_labels, num_classes)
        miou_scores.append(miou)

    mean_miou = np.mean(miou_scores)
    print(f"Mean IoU across all files: {mean_miou}")

# Example usage
#csv_file = 'test_file_list.csv'
#main(csv_file)
